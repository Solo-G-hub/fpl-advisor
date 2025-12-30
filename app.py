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
    """Fetches live team and purchase prices using transfer history."""
    base_url = "https://fantasy.premierleague.com/api/"
    try:
        static = requests.get(f"{base_url}bootstrap-static/").json()
        players_lookup = {p['id']: p['web_name'] for p in static["elements"]}
        players_now_cost = {p['id']: p['now_cost'] for p in static["elements"]}
        
        r = requests.get(f"{base_url}entry/{team_id}/event/{current_gw}/picks/")
        if r.status_code != 200:
             r = requests.get(f"{base_url}entry/{team_id}/event/{current_gw-1}/picks/")
        
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
    
    st.markdown("---")
    st.header("🧠 Decision Logic")
    min_gain_threshold = st.slider("Min XP Gain to Transfer", 0.0, 3.0, 0.75, 0.25)
    min_minutes = st.slider("Min Minutes Played (Season)", 0, 3000, 0, 90)
    include_doubtful = st.toggle("Include Doubtful (Yellow Flags)", value=True) 
    allow_hit = st.checkbox("Allow -4 Hit (+1 Extra Transfer)", value=False)
    
    st.divider()
    if st.button("🔄 Sync Prices with FPL"):
        sync_prices_to_sheets(team_id, current_gw)
    
    if 'last_sync' in st.session_state:
        st.caption(f"Last Synced: {st.session_state.last_sync}")

# --- CORE FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_fpl_data(t_id, gw, horizon, min_mins, inc_doubtful):
    base_url = "https://fantasy.premierleague.com/api/"
    try:
        static = requests.get(f"{base_url}bootstrap-static/").json()
        players = pd.DataFrame(static["elements"])
        teams = {t["id"]: t["name"] for t in static["teams"]}
        events = pd.DataFrame(static["events"])
        fixtures = pd.DataFrame(requests.get(f"{base_url}fixtures/").json())
        players["team_name"] = players["team"].map(teams)
        
        current_gw_api = int(events[events["is_current"]].iloc[0]["id"]) if not events[events["is_current"]].empty else gw
        gw_fetch = min(int(gw)-1, current_gw_api)

        history = requests.get(f"{base_url}entry/{t_id}/history/").json()
        used_chips = [c['name'] for c in history.get('chips', [])]

        r_picks = requests.get(f"{base_url}entry/{t_id}/event/{gw_fetch}/picks/")
        picks_data = r_picks.json() if r_picks.status_code == 200 else None
        if not picks_data: raise ValueError("Invalid team ID")
        
        owned_ids = [p['element'] for p in picks_data["picks"]]
        bank = picks_data["entry_history"]["bank"] / 10

        price_map = {}
        players["web_name_clean"] = players["web_name"].str.strip().str.lower()
        try:
            df_gsheet = conn.read(worksheet="Prices", ttl=0)
            if not df_gsheet.empty and 'web_name' in df_gsheet.columns:
                price_map = {str(row['web_name']).strip().lower(): row['purchase_price'] for _, row in df_gsheet.iterrows() if 'purchase_price' in row}
        except:
            pass
        
        players["current_price"] = players["now_cost"] / 10
        players["cost"] = players["current_price"]
        players["purchase_price"] = players["web_name_clean"].map(price_map).fillna(players["current_price"])

        def calc_sell(row):
            pp, cp = row['purchase_price'], row['current_price']
            return pp + 0.5 * (cp - pp) if cp > pp else cp
        
        players["selling_price"] = players.apply(calc_sell, axis=1)

        def get_fdr(team_id, start_gw, h):
            fut = fixtures[(fixtures["event"] >= start_gw) & (fixtures["event"] < start_gw + h)]
            diffs = [row["team_h_difficulty"] if row["team_h"] == team_id else row["team_a_difficulty"]
                     for _, row in fut.iterrows() if row["team_h"] == team_id or row["team_a"] == team_id]
            return sum(diffs)/len(diffs) if diffs else 3.0

        players["avg_fdr"] = players["team"].apply(lambda x: get_fdr(x, gw, horizon))
        players["base_xp"] = pd.to_numeric(players["ep_next"], errors="coerce").fillna(0)
        
        # --- XP CALCULATION WITH DOUBTFUL PENALTY ---
        # Formula: Base XP * (Fixture Multiplier) * (Status Multiplier)
        players["xp"] = players["base_xp"] * (1 + (3 - players["avg_fdr"]) * 0.1 * fdr_weight)
        players.loc[players["status"] == "d", "xp"] *= 0.75 # 25% penalty for yellow flags
        
        players["pos_name"] = players["element_type"].map({1:"GKP",2:"DEF",3:"MID",4:"FWD"})
        
        allowed_statuses = ["a", "d"] if inc_doubtful else ["a"]
        eligible_mask = (players["status"].isin(allowed_statuses)) & (players["minutes"] >= min_mins)
        final_pool = players[eligible_mask | players["id"].isin(owned_ids)].copy()
        
        return final_pool, owned_ids, bank, used_chips
    except Exception as e:
        st.error(f"FPL Error: {e}")
        return None, [], 0.0, []

def run_optimizer(players, owned_ids, budget, is_wc, allow_hit, ft_available):
    prob = pulp.LpProblem("FPL_Optimization", pulp.LpMaximize)
    s = pulp.LpVariable.dicts("squad", players.index, cat=pulp.LpBinary)  
    lineup = pulp.LpVariable.dicts("lineup", players.index, cat=pulp.LpBinary)  
    
    starters_score = pulp.lpSum([players.loc[i, 'xp'] * lineup[i] for i in players.index])
    bench_score = pulp.lpSum([players.loc[i, 'xp'] * (s[i] - lineup[i]) for i in players.index]) * 0.15
    loyalty = 0.0 if is_wc else 0.5
    loyalty_score = pulp.lpSum([loyalty for i in players.index if players.loc[i, 'id'] in owned_ids and s[i] == 1])
    
    prob += starters_score + bench_score + loyalty_score
    prob += pulp.lpSum([s[i] for i in players.index]) == 15
    prob += pulp.lpSum([players.loc[i, 'cost'] * s[i] for i in players.index]) <= budget
    
    for p_id, count in {1: 2, 2: 5, 3: 5, 4: 3}.items():
        prob += pulp.lpSum([s[i] for i in players.index if players.loc[i, 'element_type'] == p_id]) == count
    
    for t in players.team_name.unique():
        prob += pulp.lpSum([s[i] for i in players.index if players.loc[i, 'team_name'] == t]) <= 3
        
    if not is_wc:
        total_transfers = ft_available + (1 if allow_hit else 0)
        limit = 15 - total_transfers
        prob += pulp.lpSum([s[i] for i in players.index if players.loc[i,'id'] in owned_ids]) >= limit

    prob += pulp.lpSum([lineup[i] for i in players.index]) == 11
    for i in players.index: 
        prob += lineup[i] <= s[i]  
    
    prob += pulp.lpSum([lineup[i] for i in players.index if players.loc[i, 'element_type'] == 1]) == 1
    prob += pulp.lpSum([lineup[i] for i in players.index if players.loc[i, 'element_type'] == 2]) >= 3
    prob += pulp.lpSum([lineup[i] for i in players.index if players.loc[i, 'element_type'] == 4]) >= 1
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    res = players.loc[[i for i in players.index if s[i].varValue == 1]].copy()
    res['Status_Icon'] = ["⚽ START" if lineup[i].varValue == 1 else "🪑 BENCH" for i in res.index]
    
    starters_df = res[res['Status_Icon'] == "⚽ START"].sort_values(by='xp', ascending=False)
    cap_name = starters_df.iloc[0]['web_name']
    vc_name = starters_df.iloc[1]['web_name']
    
    res['sort_rank'] = 0
    res.loc[(res['Status_Icon'] == "🪑 BENCH") & (res['element_type'] == 1), 'sort_rank'] = 1
    res.loc[(res['Status_Icon'] == "🪑 BENCH") & (res['element_type'] != 1), 'sort_rank'] = 2
    res = res.sort_values(by=['sort_rank', 'xp'], ascending=[True, False])
    return res, cap_name, vc_name

# --- MAIN APP EXECUTION ---
players, owned_ids, live_bank, used_chips = get_fpl_data(team_id, current_gw, horizon, min_minutes, include_doubtful)

if players is not None:
    if 'squad_ids' not in st.session_state:
        st.session_state.squad_ids = owned_ids

    real_sell_value = players.loc[players['id'].isin(owned_ids), 'selling_price'].sum()
    initial_wealth = real_sell_value + live_bank

    current_df = players[players['id'].isin(st.session_state.squad_ids)]
    is_sim = st.session_state.squad_ids != owned_ids
    
    current_sell_value = current_df['selling_price'].sum()
    current_bank = initial_wealth - current_df['cost'].sum() if is_sim else live_bank
    dynamic_wealth = current_sell_value + current_bank

    tab1, tab2 = st.tabs(["🚀 Transfer Optimizer", "📋 My Squad & Prices"])

    with tab1:
        st.subheader("💰 Dynamic Financial Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Dynamic Wealth", f"£{dynamic_wealth:.1f}m")
        m2.metric("Live Bank", f"£{live_bank:.2f}m")
        m3.metric("Sim Bank", f"£{current_bank:.2f}m", delta=round(current_bank - live_bank, 2) if is_sim else None)
        
        st.divider()
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 Optimize Wildcard"):
                sq, cap, vc = run_optimizer(players, owned_ids, initial_wealth - buffer, True, allow_hit, 15)
                st.session_state.squad_ids = sq['id'].tolist()
                st.rerun()
        with col_btn2:
            if st.button("🔄 Suggest Transfer Strategy"):
                sq, cap, vc = run_optimizer(players, owned_ids, initial_wealth - buffer, False, allow_hit, ft_available)
                st.session_state.squad_ids = sq['id'].tolist()
                st.rerun()

    if is_sim:
        is_wildcard = (len(set(st.session_state.squad_ids) - set(owned_ids)) > ft_available + 1)
        res_sq, cap, vc = run_optimizer(players, owned_ids, initial_wealth - buffer, is_wildcard, allow_hit, ft_available)
        
        with tab1:
            # --- RISK WARNING BANNER ---
            starting_doubtful = res_sq[(res_sq['Status_Icon'] == "⚽ START") & (res_sq['status'] == "d")]
            if len(starting_doubtful) >= 2:
                st.warning(f"⚠️ **High Risk Alert:** Your optimized starting lineup contains **{len(starting_doubtful)}** doubtful players. Note: XP for doubtful players has been auto-reduced by 25%.")
            
            st.subheader("🔁 Recommended Moves")
            old_set, new_set = set(owned_ids), set(res_sq['id'].tolist())
            out_p, in_p = players[players['id'].isin(old_set - new_set)], players[players['id'].isin(new_set - old_set)]
            
            c_out, c_in = st.columns(2)
            for _, p in out_p.iterrows(): c_out.error(f"OUT: {p['web_name']}")
            for _, p in in_p.iterrows(): 
                status_note = " ⚠️" if p['status'] == "d" else ""
                c_in.success(f"IN: {p['web_name']}{status_note}")
            
            st.divider()
            res_sq.loc[res_sq['web_name'] == cap, 'Status_Icon'] = "👑 CAPTAIN"
            res_sq.loc[res_sq['web_name'] == vc, 'Status_Icon'] = "🥈 VICE-CAP"
            
            # Show XP adjusted for status in table
            st.table(res_sq[['Status_Icon', 'pos_name', 'team_name', 'web_name', 'xp', 'status']])

    with tab2:
        if is_sim:
            if st.button("↩️ Reset to My Real Squad"):
                st.session_state.squad_ids = owned_ids
                st.rerun()
        
        df_view = players[players['id'].isin(st.session_state.squad_ids)].copy()
        st.dataframe(df_view[['web_name','team_name','pos_name','purchase_price','current_price','selling_price','xp','minutes','status']]
                     .style.background_gradient(subset=['xp'], cmap='RdYlGn')
                     .format({'xp':'{:.2f}', 'selling_price':'£{:.1f}m', 'current_price':'£{:.1f}m', 'purchase_price':'£{:.1f}m'}), use_container_width=True)

else:
    st.warning("Please enter your Team ID in the sidebar to begin.")
