import streamlit as st
from streamlit_gsheets import GSheetsConnection
import requests
import pandas as pd
import pulp
import math
from datetime import datetime

# --- APP SETUP ---
st.set_page_config(page_title="FPL Tactical Advisor", layout="wide")
st.title("⚽ FPL Tactical Advisor: Second Half Pro")

# --- GOOGLE SHEETS CONNECTION ---
conn = st.connection("gsheets", type=GSheetsConnection)

# --- AUTO-GAMEWEEK INITIALIZATION ---
try:
    static_init = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    events_init = pd.DataFrame(static_init["events"])
    next_gw_auto = int(events_init[events_init["is_next"] == True].iloc[0]["id"]) if not events_init[events_init["is_next"] == True].empty else 38
except:
    next_gw_auto = 19


def sync_prices_to_sheets(team_id, current_gw):
    """Fetches live team and purchase prices using transfer history for accuracy."""
    base_url = "https://fantasy.premierleague.com/api/"
    try:
        static = requests.get(f"{base_url}bootstrap-static/").json()
        players_lookup = {p['id']: p['web_name'] for p in static["elements"]}
        players_now_cost = {p['id']: p['now_cost'] for p in static["elements"]}

        r = requests.get(f"{base_url}entry/{team_id}/event/{current_gw}/picks/")
        if r.status_code != 200:
            r = requests.get(f"{base_url}entry/{team_id}/event/{current_gw - 1}/picks/")

        if r.status_code == 200:
            picks_list = r.json().get('picks', [])
            transfers_r = requests.get(f"{base_url}entry/{team_id}/transfers/")
            transfer_data = transfers_r.json() if transfers_r.status_code == 200 else []

            history_prices = {}
            for t in sorted(transfer_data, key=lambda x: x['time']):
                history_prices[t['element_in']] = t['element_in_cost']

            rows = []
            for p in picks_list:
                p_id = p.get('element')
                name = players_lookup.get(p_id, "Unknown")
                raw_price = history_prices.get(p_id)
                if not raw_price:
                    raw_price = p.get('purchase_price', 0)
                if raw_price == 0:
                    raw_price = players_now_cost.get(p_id, 0)
                rows.append({"web_name": name, "purchase_price": raw_price / 10.0})

            new_data = pd.DataFrame(rows)
            conn.update(worksheet="Prices", data=new_data)
            st.session_state.last_sync = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.cache_data.clear()
            st.success("✅ Prices & Transfers Synced!")
            st.rerun()
        else:
            st.error(f"❌ Sync failed. FPL API status: {r.status_code}")
    except Exception as e:
        st.error(f"Sync Error: {e}")


# ─────────────────────────────────────────────
# IMPROVEMENT 1: Pre-compute FDR as a dict
# ─────────────────────────────────────────────
def build_fdr_lookup(fixtures, start_gw, horizon):
    """
    Returns a dict: team_id -> avg_fdr over [start_gw, start_gw+horizon).
    Vectorised – called once instead of per-player.
    """
    fut = fixtures[(fixtures["event"] >= start_gw) & (fixtures["event"] < start_gw + horizon)].copy()
    records = []
    for _, row in fut.iterrows():
        records.append({"team": row["team_h"], "difficulty": row["team_h_difficulty"]})
        records.append({"team": row["team_a"], "difficulty": row["team_a_difficulty"]})
    if not records:
        return {}
    df = pd.DataFrame(records)
    return df.groupby("team")["difficulty"].mean().to_dict()


def build_weekly_fixture_lookup(fixtures, start_gw, horizon):
    """
    Returns dict: (team_id, gw_offset) -> list of difficulty values.
    Used by calibrate_horizon_xp for vectorised-friendly access.
    """
    lookup = {}
    for offset in range(horizon):
        gw = start_gw + offset
        week_fix = fixtures[fixtures["event"] == gw]
        for _, row in week_fix.iterrows():
            for side, diff in [("team_h", row["team_h_difficulty"]), ("team_a", row["team_a_difficulty"])]:
                tid = row[side]
                key = (tid, offset)
                lookup.setdefault(key, []).append(diff)
    return lookup


# ─────────────────────────────────────────────
# IMPROVEMENT 2: Detect blank/DGW from API data
# ─────────────────────────────────────────────
def detect_blank_dgw_events(fixtures, events):
    """
    Returns two dicts: {gw: [team_ids_missing]}, {gw: [team_ids_with_double]}
    Uses fixture data rather than hardcoded dates.
    """
    all_team_ids = set(fixtures["team_h"].unique()) | set(fixtures["team_a"].unique())
    blanks, doubles = {}, {}

    future_events = events[events["id"] > 0]["id"].tolist()
    for gw in future_events:
        gw_fix = fixtures[fixtures["event"] == gw]
        playing = set(gw_fix["team_h"].tolist()) | set(gw_fix["team_a"].tolist())
        missing = all_team_ids - playing
        double_teams = []
        counts = gw_fix["team_h"].value_counts().add(gw_fix["team_a"].value_counts(), fill_value=0)
        for t, c in counts.items():
            if c >= 2:
                double_teams.append(int(t))
        if missing:
            blanks[int(gw)] = [int(t) for t in missing]
        if double_teams:
            doubles[int(gw)] = double_teams
    return blanks, doubles


# ─────────────────────────────────────────────
# IMPROVEMENT 3: Price rise/fall alert helper
# ─────────────────────────────────────────────
def get_price_change_alerts(players):
    """
    Returns players where transfers_in_event >> transfers_out_event (rise risk)
    or vice versa (fall risk). Uses FPL's own transfer data.
    """
    df = players.copy()
    df["net_transfers"] = pd.to_numeric(df.get("transfers_in_event", 0), errors="coerce").fillna(0) - \
                          pd.to_numeric(df.get("transfers_out_event", 0), errors="coerce").fillna(0)
    rising = df[df["net_transfers"] > 50000].nlargest(5, "net_transfers")[
        ["web_name", "team_name", "pos_name", "current_price", "net_transfers"]]
    falling = df[df["net_transfers"] < -50000].nsmallest(5, "net_transfers")[
        ["web_name", "team_name", "pos_name", "current_price", "net_transfers"]]
    return rising, falling


# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    team_id = st.number_input("Enter FPL Team ID", value=5816864, step=1)
    current_gw = st.number_input("Target Gameweek", value=next_gw_auto, step=1)
    buffer = st.number_input("Safety Buffer (m)", min_value=0.0, max_value=2.0, value=0.2, step=0.1)

    st.markdown("---")
    st.header("📈 Strategy")
    ft_available = st.slider("Free Transfers Available", 1, 5, 1)
    horizon = st.slider("Planning Horizon (Weeks)", 1, 8, 5)
    fdr_weight = st.slider("Fixture Difficulty Weight", 0.0, 1.0, 0.5)

    st.subheader("🧪 Decay Rates (Horizon)")
    att_decay = st.slider("Attacker Decay (MID/FWD)", 0.5, 1.0, 0.9, 0.05)
    def_decay = st.slider("Defender Decay (GKP/DEF)", 0.5, 1.0, 0.75, 0.05)

    st.markdown("---")
    st.header("🧠 Decision Logic")
    min_gain_threshold = st.slider("Min XP Gain to Transfer", 0.0, 3.0, 0.75, 0.25)
    allow_hit = st.checkbox("Allow -4 Hit (+1 Extra Transfer)", value=False)

    # ─────────────────────────────────────────────
    # IMPROVEMENT 4: Pin/Exclude players
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.header("📌 Pin / Exclude Players")
    st.caption("Type exact web_names (comma-separated)")
    pinned_raw = st.text_input("Pin (keep in squad)", placeholder="e.g. Salah, Haaland")
    excluded_raw = st.text_input("Exclude (block from squad)", placeholder="e.g. Isak")
    pinned_names = [n.strip().lower() for n in pinned_raw.split(",") if n.strip()]
    excluded_names = [n.strip().lower() for n in excluded_raw.split(",") if n.strip()]

    st.divider()
    if st.button("🔄 Sync Prices with FPL"):
        sync_prices_to_sheets(team_id, current_gw)

    if 'last_sync' in st.session_state:
        st.caption(f"Last Synced: {st.session_state.last_sync}")


# --- CORE DATA FETCH ---
@st.cache_data(ttl=3600)
def get_fpl_data(t_id, gw, horizon, att_decay, def_decay, fdr_weight):
    base_url = "https://fantasy.premierleague.com/api/"
    try:
        static = requests.get(f"{base_url}bootstrap-static/").json()
        players = pd.DataFrame(static["elements"])
        teams = {t["id"]: t["name"] for t in static["teams"]}
        events = pd.DataFrame(static["events"])
        fixtures_raw = requests.get(f"{base_url}fixtures/").json()
        fixtures = pd.DataFrame(fixtures_raw)
        players["team_name"] = players["team"].map(teams)

        # Current GW fixture counts (for DGW alerts)
        target_fixtures = fixtures[fixtures["event"] == gw]
        fixture_counts = (
            target_fixtures["team_h"].value_counts()
            .add(target_fixtures["team_a"].value_counts(), fill_value=0)
        ).to_dict()
        players["gw_fixtures"] = players["team"].map(fixture_counts).fillna(0).astype(int)

        current_gw_api = int(events[events["is_current"]].iloc[0]["id"]) if not events[events["is_current"]].empty else gw
        gw_fetch = min(int(gw) - 1, current_gw_api)

        history = requests.get(f"{base_url}entry/{t_id}/history/").json()
        used_chips = [c['name'] for c in history.get('chips', []) if c['event'] >= 20]

        # ── IMPROVEMENT: also return full chip history for ROI tracker ──
        all_chips = history.get('chips', [])

        r_picks = requests.get(f"{base_url}entry/{t_id}/event/{gw_fetch}/picks/")
        picks_data = r_picks.json() if r_picks.status_code == 200 else None

        if not picks_data:
            raise ValueError("Invalid team ID or no data for this gameweek")

        owned_ids = [p['element'] for p in picks_data["picks"]]
        bank = picks_data["entry_history"]["bank"] / 10

        # ── IMPROVEMENT: fetch live points for current GW ──
        live_points_map = {}
        try:
            live_r = requests.get(f"{base_url}event/{current_gw_api}/live/")
            if live_r.status_code == 200:
                for elem in live_r.json().get("elements", []):
                    live_points_map[elem["id"]] = elem["stats"].get("total_points", 0)
        except:
            pass

        # ── IMPROVEMENT: Transfer history for ROI tracker ──
        transfers_r = requests.get(f"{base_url}entry/{t_id}/transfers/")
        transfer_history = transfers_r.json() if transfers_r.status_code == 200 else []

        price_map = {}
        players["web_name_clean"] = players["web_name"].str.strip().str.lower()
        try:
            df_gsheet = conn.read(worksheet="Prices", ttl=0)
            if not df_gsheet.empty and 'web_name' in df_gsheet.columns:
                price_map = {
                    str(row['web_name']).strip().lower(): row['purchase_price']
                    for _, row in df_gsheet.iterrows() if 'purchase_price' in row
                }
        except:
            pass

        players["current_price"] = players["now_cost"] / 10
        players["cost"] = players["current_price"]
        players["purchase_price"] = players["web_name_clean"].map(price_map).fillna(players["current_price"])

        def calc_sell(row):
            pp, cp = row['purchase_price'], row['current_price']
            return pp + 0.5 * (cp - pp) if cp > pp else cp

        players["selling_price"] = players.apply(calc_sell, axis=1)

        # ── IMPROVEMENT: Vectorised FDR (pre-computed dict) ──
        fdr_lookup = build_fdr_lookup(fixtures, gw, horizon)
        players["avg_fdr"] = players["team"].map(fdr_lookup).fillna(3.0)

        players["base_xp"] = pd.to_numeric(players["ep_next"], errors="coerce").fillna(0)

        # ── IMPROVEMENT: Weekly fixture lookup dict for horizon XP ──
        weekly_lookup = build_weekly_fixture_lookup(fixtures, gw, horizon)

        def calibrate_horizon_xp(row):
            decay = def_decay if row["element_type"] in [1, 2] else att_decay
            pos_sensitivity = 1.5 if row["element_type"] in [1, 2] else 0.7
            total_projected_xp = 0
            for week_offset in range(horizon):
                diffs = weekly_lookup.get((row["team"], week_offset), [])
                gw_xp_acc = sum(
                    row["base_xp"] * (1 + (3 - d) * 0.1 * fdr_weight * pos_sensitivity)
                    for d in diffs
                )
                total_projected_xp += gw_xp_acc * (decay ** week_offset)
            return total_projected_xp

        players["xp"] = players.apply(calibrate_horizon_xp, axis=1)
        players["pos_name"] = players["element_type"].map({1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"})

        # Live points column
        players["live_pts"] = players["id"].map(live_points_map).fillna(0).astype(int)

        # ── IMPROVEMENT: Detect blank/DGW dynamically ──
        blanks, doubles = detect_blank_dgw_events(fixtures, events)

        return players, owned_ids, bank, used_chips, all_chips, transfer_history, blanks, doubles

    except Exception as e:
        st.error(f"FPL Error: {e}")
        return None, [], 0.0, [], [], [], {}, {}


# --- OPTIMIZER ---
def run_optimizer(players, owned_ids, budget, is_wc, allow_hit, ft_available,
                  pinned_ids=None, excluded_ids=None):
    pinned_ids = pinned_ids or []
    excluded_ids = excluded_ids or []

    # Filter out excluded players
    eligible = players[~players['id'].isin(excluded_ids)].copy()

    prob = pulp.LpProblem("FPL_Optimization", pulp.LpMaximize)

    s = pulp.LpVariable.dicts("squad", eligible.index, cat=pulp.LpBinary)
    lineup = pulp.LpVariable.dicts("lineup", eligible.index, cat=pulp.LpBinary)
    captain = pulp.LpVariable.dicts("captain", eligible.index, cat=pulp.LpBinary)

    if not is_wc:
        transfer = pulp.LpVariable.dicts("transfer", eligible.index, cat=pulp.LpBinary)

    # DGW-adjusted captain XP
    captain_xp_values = {}
    for i in eligible.index:
        gw_fix = eligible.loc[i, 'gw_fixtures']
        bxp = eligible.loc[i, 'base_xp']
        captain_xp_values[i] = bxp + (bxp * 0.7) if gw_fix > 1 else bxp

    starters_score = pulp.lpSum([eligible.loc[i, 'xp'] * lineup[i] for i in eligible.index])
    captain_bonus = pulp.lpSum([captain_xp_values[i] * captain[i] for i in eligible.index])
    bench_score = pulp.lpSum([eligible.loc[i, 'xp'] * (s[i] - lineup[i]) for i in eligible.index]) * 0.15

    if is_wc:
        transfer_penalty = 0
        loyalty_score = 0
    else:
        for i in eligible.index:
            if eligible.loc[i, 'id'] in owned_ids:
                prob += transfer[i] >= 1 - s[i]
            else:
                prob += transfer[i] >= s[i]

        total_transfers = pulp.lpSum([transfer[i] for i in eligible.index]) / 2
        num_hits = pulp.LpVariable("num_hits", lowBound=0, cat=pulp.LpInteger)
        prob += num_hits >= total_transfers - ft_available
        transfer_penalty = num_hits * 4.0
        loyalty_score = pulp.lpSum(
            [0.5 * s[i] for i in eligible.index if eligible.loc[i, 'id'] in owned_ids]
        )

    prob += starters_score + captain_bonus + bench_score + (loyalty_score if not is_wc else 0) - (
        transfer_penalty if not is_wc else 0)

    prob += pulp.lpSum([s[i] for i in eligible.index]) == 15
    prob += pulp.lpSum([eligible.loc[i, 'cost'] * s[i] for i in eligible.index]) <= budget

    for p_id, count in {1: 2, 2: 5, 3: 5, 4: 3}.items():
        prob += pulp.lpSum([s[i] for i in eligible.index if eligible.loc[i, 'element_type'] == p_id]) == count

    for t in eligible.team_name.unique():
        prob += pulp.lpSum([s[i] for i in eligible.index if eligible.loc[i, 'team_name'] == t]) <= 3

    prob += pulp.lpSum([lineup[i] for i in eligible.index]) == 11
    prob += pulp.lpSum([captain[i] for i in eligible.index]) == 1

    for i in eligible.index:
        prob += lineup[i] <= s[i]
        prob += captain[i] <= lineup[i]

    prob += pulp.lpSum([lineup[i] for i in eligible.index if eligible.loc[i, 'element_type'] == 1]) == 1
    prob += pulp.lpSum([lineup[i] for i in eligible.index if eligible.loc[i, 'element_type'] == 2]) >= 3
    prob += pulp.lpSum([lineup[i] for i in eligible.index if eligible.loc[i, 'element_type'] == 4]) >= 1

    # ── IMPROVEMENT: Force pinned players into squad ──
    for i in eligible.index:
        if eligible.loc[i, 'id'] in pinned_ids:
            prob += s[i] == 1

    if not is_wc:
        max_transfers = ft_available + (5 if allow_hit else 0)
        prob += (pulp.lpSum([transfer[i] for i in eligible.index]) / 2) <= max_transfers

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    res = eligible.loc[[i for i in eligible.index if s[i].varValue == 1]].copy()
    cap_id = [i for i in eligible.index if captain[i].varValue == 1][0]
    cap_name = eligible.loc[cap_id, 'web_name']

    starters_ids = [i for i in eligible.index if lineup[i].varValue == 1]
    vc_options = res[res.index.isin(starters_ids) & (res.index != cap_id)]
    vc_row = vc_options.sort_values(by='xp', ascending=False).iloc[0]
    vc_id = vc_row.name
    vc_name = vc_row['web_name']

    res['Status'] = ["⚽ START" if lineup[i].varValue == 1 else "🪑 BENCH" for i in res.index]
    res.loc[cap_id, 'Status'] = "👑 CAPTAIN"
    res.loc[vc_id, 'Status'] = "🥈 VICE-CAP"

    res['sort_rank'] = 0
    res.loc[res['Status'] == "👑 CAPTAIN", 'sort_rank'] = -1
    res.loc[res['Status'] == "🥈 VICE-CAP", 'sort_rank'] = -0.5
    res.loc[(res['Status'] == "🪑 BENCH") & (res['element_type'] == 1), 'sort_rank'] = 1
    res.loc[(res['Status'] == "🪑 BENCH") & (res['element_type'] != 1), 'sort_rank'] = 2
    res = res.sort_values(by=['sort_rank', 'xp'], ascending=[True, False])

    return res, cap_name, vc_name


# ─────────────────────────────────────────────
# IMPROVEMENT 5: Formation pitch visualizer
# ─────────────────────────────────────────────
def render_pitch(squad_df):
    """Renders an SVG football pitch with players positioned by formation."""
    starters = squad_df[squad_df['Status'].str.contains('⚽|👑|🥈')].copy()
    bench = squad_df[squad_df['Status'] == "🪑 BENCH"].copy()

    gkp = starters[starters['element_type'] == 1]
    defs = starters[starters['element_type'] == 2]
    mids = starters[starters['element_type'] == 3]
    fwds = starters[starters['element_type'] == 4]

    W, H = 500, 680
    pitch_green = "#2d7a3a"
    line_col = "rgba(255,255,255,0.5)"
    text_col = "#ffffff"

    def player_svg(x, y, name, status, xp_val):
        short = name[:10]
        if "👑" in status:
            badge_color = "#FFD700"
            badge = "C"
        elif "🥈" in status:
            badge_color = "#C0C0C0"
            badge = "V"
        else:
            badge_color = "#1a73e8"
            badge = ""

        badge_svg = ""
        if badge:
            badge_svg = f"""
            <circle cx="{x + 18}" cy="{y - 18}" r="8" fill="{badge_color}" stroke="white" stroke-width="1"/>
            <text x="{x + 18}" y="{y - 14}" text-anchor="middle" font-size="9" font-weight="bold" fill="black">{badge}</text>
            """

        return f"""
        <circle cx="{x}" cy="{y}" r="22" fill="{badge_color if badge else '#1a73e8'}" stroke="white" stroke-width="2"/>
        {badge_svg}
        <text x="{x}" y="{y + 4}" text-anchor="middle" font-size="9" font-weight="bold" fill="{text_col}">{short}</text>
        <text x="{x}" y="{y + 34}" text-anchor="middle" font-size="8" fill="{text_col}" opacity="0.85">{xp_val:.1f}xp</text>
        """

    def row_positions(n, y, width=W):
        if n == 0:
            return []
        spacing = width / (n + 1)
        return [(spacing * (i + 1), y) for i in range(n)]

    svg_parts = [f"""
    <svg viewBox="0 0 {W} {H + 80}" xmlns="http://www.w3.org/2000/svg" style="background:{pitch_green};border-radius:12px;width:100%">
    <!-- Pitch markings -->
    <rect x="30" y="10" width="{W-60}" height="{H-20}" rx="4" fill="none" stroke="{line_col}" stroke-width="1.5"/>
    <line x1="30" y1="{H//2}" x2="{W-30}" y2="{H//2}" stroke="{line_col}" stroke-width="1"/>
    <circle cx="{W//2}" cy="{H//2}" r="50" fill="none" stroke="{line_col}" stroke-width="1"/>
    <rect x="{W//2 - 80}" y="10" width="160" height="60" fill="none" stroke="{line_col}" stroke-width="1"/>
    <rect x="{W//2 - 80}" y="{H-70}" width="160" height="60" fill="none" stroke="{line_col}" stroke-width="1"/>
    """]

    rows = [
        (list(gkp.iterrows()), 580),
        (list(defs.iterrows()), 460),
        (list(mids.iterrows()), 310),
        (list(fwds.iterrows()), 175),
    ]

    for players_list, y in rows:
        positions = row_positions(len(players_list), y)
        for (_, p), (x, py) in zip(players_list, positions):
            svg_parts.append(player_svg(x, py, p['web_name'], p['Status'], p['xp']))

    # Bench strip
    svg_parts.append(f'<rect x="0" y="{H + 5}" width="{W}" height="75" fill="rgba(0,0,0,0.35)"/>')
    svg_parts.append(f'<text x="{W//2}" y="{H + 22}" text-anchor="middle" font-size="10" fill="{text_col}" opacity="0.7">BENCH</text>')

    bench_positions = row_positions(min(len(bench), 4), H + 50)
    for (_, p), (x, y) in zip(bench.iterrows(), bench_positions):
        svg_parts.append(player_svg(x, y, p['web_name'], p['Status'], p['xp']))

    svg_parts.append("</svg>")
    st.markdown("".join(svg_parts), unsafe_allow_html=True)


# ─────────────────────────────────────────────
# IMPROVEMENT 6: Transfer ROI tracker
# ─────────────────────────────────────────────
def render_transfer_roi(transfer_history, players):
    """Shows a table of past transfers with points gained/lost."""
    if not transfer_history:
        st.info("No transfer history found.")
        return

    player_pts = dict(zip(players['id'], players.get('total_points', players['xp'])))
    player_names = dict(zip(players['id'], players['web_name']))

    rows = []
    for t in transfer_history[-20:]:  # last 20 transfers
        p_in = t.get('element_in')
        p_out = t.get('element_out')
        rows.append({
            "GW": t.get('event', '?'),
            "IN": player_names.get(p_in, str(p_in)),
            "Cost (IN)": f"£{t.get('element_in_cost', 0) / 10:.1f}m",
            "OUT": player_names.get(p_out, str(p_out)),
            "Sold (OUT)": f"£{t.get('element_out_cost', 0) / 10:.1f}m",
        })

    df = pd.DataFrame(rows).sort_values("GW", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# IMPROVEMENT 7: Dynamic chip strategy advisor
# ─────────────────────────────────────────────
def render_chip_advice(current_gw, used_chips, blanks, doubles):
    st.subheader("💡 Dynamic Chip Strategy Advisor")

    upcoming_blanks = {gw: t for gw, t in blanks.items() if gw >= current_gw and len(t) >= 4}
    upcoming_doubles = {gw: t for gw, t in doubles.items() if gw >= current_gw and len(t) >= 2}

    col1, col2 = st.columns(2)
    with col1:
        if upcoming_blanks:
            next_blank = min(upcoming_blanks.keys())
            st.error(f"🚨 **Blank GW{next_blank}**: {len(upcoming_blanks[next_blank])} teams missing. "
                     f"{'Free Hit available ✅' if 'freehit' not in used_chips else 'Free Hit used ❌'}")
        else:
            st.success("✅ No upcoming blank gameweeks detected.")
    with col2:
        if upcoming_doubles:
            next_dgw = min(upcoming_doubles.keys())
            st.success(f"🚀 **Double GW{next_dgw}**: {len(upcoming_doubles[next_dgw])} teams playing twice. "
                       f"{'Bench Boost available ✅' if 'bboost' not in used_chips else 'Bench Boost used ❌'}")
        else:
            st.info("No upcoming double gameweeks detected yet.")

    # Wildcard window advice
    if 'wildcard' not in used_chips and upcoming_blanks:
        next_blank = min(upcoming_blanks.keys())
        wc_target = next_blank - 2
        if current_gw <= wc_target:
            st.warning(f"🃏 **Wildcard Window:** Consider wildcarding in GW{wc_target} "
                       f"to build a squad for the blank in GW{next_blank}.")


# ─── MAIN APP ───
result = get_fpl_data(team_id, current_gw, horizon, att_decay, def_decay, fdr_weight)
players, owned_ids, live_bank, used_chips, all_chips, transfer_history, blanks, doubles = result

if players is not None:
    # Resolve pin/exclude names to IDs
    players["web_name_clean"] = players["web_name"].str.strip().str.lower()
    pinned_ids = players[players["web_name_clean"].isin(pinned_names)]["id"].tolist()
    excluded_ids = players[players["web_name_clean"].isin(excluded_names)]["id"].tolist()

    if 'squad_ids' not in st.session_state:
        st.session_state.squad_ids = owned_ids

    real_sell_value = players.loc[players['id'].isin(owned_ids), 'selling_price'].sum()
    initial_wealth = real_sell_value + live_bank

    current_df = players[players['id'].isin(st.session_state.squad_ids)]
    is_sim = st.session_state.squad_ids != owned_ids

    current_sell_value = current_df['selling_price'].sum()

    if is_sim:
        total_team_cost = sum(
            p['selling_price'] if p['id'] in owned_ids else p['current_price']
            for _, p in current_df.iterrows()
        )
        current_bank = initial_wealth - total_team_cost
    else:
        current_bank = live_bank

    dynamic_wealth = current_sell_value + current_bank

    # ── Tabs: added Live Points + Transfer History ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Transfer Optimizer",
        "📋 My Squad & Prices",
        "📡 Live Points",
        "📜 Transfer History"
    ])

    # ─── TAB 1: OPTIMIZER ───
    with tab1:
        total_team_xp = current_df['xp'].sum()
        avg_team_fdr = current_df['avg_fdr'].mean()

        if total_team_xp < (45 * horizon * 0.7) or avg_team_fdr > 3.7:
            st.info("🔔 **Chip Recommendation Available:** Check Tab 2 for strategy advice.")

        st.subheader("💰 Dynamic Financial Summary")
        m_wealth, m_live, m_sim = st.columns(3)
        m_wealth.metric("Dynamic Wealth", f"£{dynamic_wealth:.1f}m")
        m_live.metric("Live Bank (FPL)", f"£{live_bank:.2f}m")
        m_sim.metric("Remaining Bank (Sim)", f"£{current_bank:.2f}m",
                     delta=round(current_bank - live_bank, 2) if is_sim else None)

        # ── Price alerts ──
        rising, falling = get_price_change_alerts(players)
        if not rising.empty or not falling.empty:
            with st.expander("📈 Price Change Alerts (based on net transfers)"):
                c_rise, c_fall = st.columns(2)
                with c_rise:
                    st.markdown("**🟢 Rising (buy before price increases)**")
                    st.dataframe(rising.rename(columns={"net_transfers": "Net Transfers In"}),
                                 use_container_width=True, hide_index=True)
                with c_fall:
                    st.markdown("**🔴 Falling (sell before price drops)**")
                    st.dataframe(falling.rename(columns={"net_transfers": "Net Transfers Out"}),
                                 use_container_width=True, hide_index=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚀 Optimize Wildcard"):
                sq, cap, vc = run_optimizer(players, owned_ids, initial_wealth - buffer, True,
                                            allow_hit, 15, pinned_ids, excluded_ids)
                st.session_state.squad_ids = sq['id'].tolist()
                st.rerun()
        with c2:
            if st.button("🔄 Suggest Transfer Strategy"):
                sq, cap, vc = run_optimizer(players, owned_ids, initial_wealth - buffer, False,
                                            allow_hit, ft_available, pinned_ids, excluded_ids)
                st.session_state.squad_ids = sq['id'].tolist()
                st.rerun()

        if is_sim:
            is_wildcard = (len(set(st.session_state.squad_ids) - set(owned_ids)) > ft_available + 1)
            res_sq, cap, vc = run_optimizer(players, owned_ids, initial_wealth - buffer,
                                            is_wc=is_wildcard, allow_hit=allow_hit,
                                            ft_available=ft_available,
                                            pinned_ids=pinned_ids, excluded_ids=excluded_ids)

            st.subheader("🔁 Recommended Moves")
            old_set = set(owned_ids)
            new_set = set(res_sq['id'].tolist())
            out_players = players[players['id'].isin(old_set - new_set)]
            in_players = players[players['id'].isin(new_set - old_set)]

            if not in_players.empty:
                col_out, col_in = st.columns(2)
                with col_out:
                    for _, p in out_players.iterrows():
                        st.error(f"OUT: {p['web_name']} ({p['team_name']})")
                with col_in:
                    for _, p in in_players.iterrows():
                        st.success(f"IN: {p['web_name']} ({p['team_name']})")

                current_total_xp = players[players['id'].isin(owned_ids)]['xp'].sum()
                new_total_xp = res_sq['xp'].sum()
                net_gain = new_total_xp - current_total_xp

                if net_gain < min_gain_threshold and not is_wildcard:
                    st.warning(f"⚠️ **Marginal Gain:** +{net_gain:.2f} XP. Threshold is {min_gain_threshold}. Consider banking!")
                else:
                    st.info(f"✨ **Strategy Value:** Squad improves by {net_gain:.2f} Total Horizon XP.")
            else:
                st.info("✅ Your current squad is mathematically optimal. No transfers needed!")

            st.divider()
            starters_xp = res_sq[res_sq['Status'].str.contains('⚽|👑|🥈')]['xp'].sum()
            bench_xp = res_sq[res_sq['Status'] == "🪑 BENCH"]['xp'].sum()
            weighted_total = starters_xp + (bench_xp * 0.15)

            st.success(
                f"Total Horizon XP: {weighted_total:.1f} (Starters: {starters_xp:.1f} + Bench Value) "
                f"| 👑 Captain: {cap} | 🥈 Vice: {vc}"
            )

            # ── IMPROVEMENT: Pitch view toggle ──
            view_mode = st.radio("View Mode", ["📋 Table", "⚽ Pitch View"], horizontal=True)
            if view_mode == "⚽ Pitch View":
                render_pitch(res_sq)
            else:
                st.table(res_sq[['Status', 'pos_name', 'team_name', 'web_name', 'xp']])

    # ─── TAB 2: MY SQUAD ───
    with tab2:
        st.subheader("🃏 Chip Strategy Advisor")
        c_chips = st.columns(4)
        chip_names = {"wildcard": "Wildcard", "freehit": "Free Hit", "3xc": "Triple Captain", "bboost": "Bench Boost"}
        for i, (internal_name, display_name) in enumerate(chip_names.items()):
            is_used = internal_name in used_chips
            c_chips[i].metric(display_name, "❌ Used" if is_used else "✅ Available")

        st.divider()

        # ── IMPROVEMENT: Dynamic chip advice (replaces hardcoded roadmap) ──
        render_chip_advice(current_gw, used_chips, blanks, doubles)

        st.divider()
        st.subheader("💡 Current Squad Radar")
        dgw_players = current_df[current_df['gw_fixtures'] >= 2]
        if not dgw_players.empty and 'bboost' not in used_chips:
            st.success(f"🚀 **Bench Boost Potential:** You have {len(dgw_players)} players playing twice this week!")

        top_p = current_df.nlargest(1, 'xp').iloc[0]
        if top_p['gw_fixtures'] >= 2 and '3xc' not in used_chips:
            st.info(f"👑 **Triple Captain Alert:** {top_p['web_name']} has a Double Gameweek. High ceiling detected.")

        blanks_this_gw = current_df[current_df['gw_fixtures'] == 0]
        if len(blanks_this_gw) >= 3 and 'freehit' not in used_chips:
            st.error(f"🃏 **Free Hit Advice:** {len(blanks_this_gw)} players have NO fixture. Consider a Free Hit.")

        st.divider()
        if is_sim:
            st.warning("⚠️ Showing Simulated Squad")
            if st.button("↩️ Reset to My Real Squad"):
                st.session_state.squad_ids = owned_ids
                st.rerun()
        else:
            st.success("✅ Showing Your Current Squad")

        df_view = players[players['id'].isin(st.session_state.squad_ids)].copy()
        st.dataframe(
            df_view[['web_name', 'team_name', 'pos_name', 'purchase_price',
                      'current_price', 'selling_price', 'xp', 'gw_fixtures', 'avg_fdr']]
            .style.background_gradient(subset=['avg_fdr'], cmap='RdYlGn_r')
            .format({
                'xp': '{:.2f}',
                'selling_price': '£{:.1f}m',
                'current_price': '£{:.1f}m',
                'purchase_price': '£{:.1f}m'
            }),
            use_container_width=True
        )

        st.divider()
        m_val, m_rem = st.columns(2)
        m_val.metric("Squad Sell Value", f"£{current_sell_value:.1f}m")
        m_rem.metric("Remaining Bank", f"£{current_bank:.2f}m")

    # ─── TAB 3: LIVE POINTS ───
    with tab3:
        st.subheader("📡 Live GW Points Tracker")
        live_squad = players[players['id'].isin(st.session_state.squad_ids)].copy()
        live_squad = live_squad.sort_values('live_pts', ascending=False)

        if live_squad['live_pts'].sum() == 0:
            st.info("Live points data not available yet (GW may not have started).")
        else:
            total_live = live_squad['live_pts'].sum()
            st.metric("Total Live Points (Squad)", int(total_live))
            st.dataframe(
                live_squad[['web_name', 'team_name', 'pos_name', 'live_pts', 'gw_fixtures']]
                .rename(columns={'live_pts': 'Live Pts'})
                .style.background_gradient(subset=['Live Pts'], cmap='Greens'),
                use_container_width=True,
                hide_index=True
            )

    # ─── TAB 4: TRANSFER HISTORY ───
    with tab4:
        st.subheader("📜 Transfer History & ROI")
        render_transfer_roi(transfer_history, players)

else:
    st.warning("Please enter your Team ID in the sidebar to begin.")
