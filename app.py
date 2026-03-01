import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.ensemble import RandomForestRegressor
import pulp
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="NextStep — Retail Location Intelligence",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0A0A0F;
    color: #E8E8F0;
}

.stApp {
    background: #0A0A0F;
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0F0F1A !important;
    border-right: 1px solid #1E1E2E;
}

[data-testid="stSidebar"] * {
    color: #E8E8F0 !important;
}

/* ── Hero header ── */
.hero {
    padding: 3rem 0 2rem 0;
    border-bottom: 1px solid #1E1E2E;
    margin-bottom: 2.5rem;
}

.hero-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #818CF8;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.3rem 0.85rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
    color: #FFFFFF;
    margin: 0 0 1rem 0;
}

.hero-title span {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    font-weight: 300;
    color: #888899;
    max-width: 560px;
    line-height: 1.6;
    margin: 0;
}

/* ── Step labels ── */
.step-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6366F1;
    margin-bottom: 0.4rem;
}

.step-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.3rem;
}

.step-desc {
    font-size: 0.9rem;
    color: #666677;
    margin-bottom: 1.5rem;
}

/* ── Upload cards ── */
.upload-card {
    background: #0F0F1A;
    border: 1px solid #1E1E2E;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}

.upload-card:hover {
    border-color: #6366F1;
}

.upload-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.2rem;
    letter-spacing: 0.02em;
}

.upload-card-desc {
    font-size: 0.78rem;
    color: #555566;
    margin-bottom: 0.8rem;
}

/* ── File uploader styling ── */
[data-testid="stFileUploader"] {
    background: #13131F !important;
    border: 1.5px dashed #2A2A3E !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #6366F1 !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #13131F !important;
    border: 1px solid #1E1E2E !important;
    border-radius: 8px !important;
    color: #E8E8F0 !important;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] > div > div {
    background: #13131F !important;
    border: 1px solid #1E1E2E !important;
    border-radius: 8px !important;
}

/* ── Slider ── */
[data-testid="stSlider"] .st-emotion-cache-1dp5vir {
    background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.75rem 2rem !important;
    height: 3.2rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 30px rgba(99, 102, 241, 0.3) !important;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 50px rgba(99, 102, 241, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #0F0F1A;
    border: 1px solid #1E1E2E;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
}

[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: #555566 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
}

/* ── Success / info / error ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}

.stSuccess {
    background: rgba(16, 185, 129, 0.1) !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    color: #10B981 !important;
}

.stInfo {
    background: rgba(99, 102, 241, 0.08) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    color: #818CF8 !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid #1E1E2E !important;
    margin: 2rem 0 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1E1E2E !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Result card ── */
.result-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}

.result-banner-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6366F1;
    margin-bottom: 0.4rem;
}

.result-banner-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #FFFFFF;
}

/* ── Sidebar logo ── */
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 0.2rem;
}

.sidebar-tagline {
    font-size: 0.75rem;
    color: #444455;
    margin-bottom: 1.5rem;
}

/* ── Column labels ── */
.col-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1E1E2E;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #13131F !important;
    color: #E8E8F0 !important;
    border: 1px solid #2A2A3E !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}

.stDownloadButton > button:hover {
    border-color: #6366F1 !important;
    color: #818CF8 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #6366F1 !important;
}

</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">📍 NextStep</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Retail Location Intelligence</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="step-label">Configuration</div>', unsafe_allow_html=True)
    max_stores = st.slider("Maximum stores to open", 1, 10, 3)

    st.divider()

    st.markdown('<div class="step-label">How it works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#555566; line-height:1.8;">
    <span style="color:#6366F1; font-weight:600;">01</span> &nbsp;Upload your 3 data files<br>
    <span style="color:#6366F1; font-weight:600;">02</span> &nbsp;Map your column names<br>
    <span style="color:#6366F1; font-weight:600;">03</span> &nbsp;Click Find Optimal Locations<br>
    <span style="color:#6366F1; font-weight:600;">04</span> &nbsp;Get results + map instantly
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem; color:#333344; line-height:1.6;">
    Powered by Random Forest ML<br>
    + Binary Integer Programming<br>
    Built by NextStep © 2025
    </div>
    """, unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Location Intelligence</div>
    <h1 class="hero-title">Find your optimal <span>store locations</span><br>in 5 minutes.</h1>
    <p class="hero-sub">Stop paying €50,000 for location consultants. Upload your data, let our AI find the best locations to maximize your revenue.</p>
</div>
""", unsafe_allow_html=True)

# ── Step 1: Upload ─────────────────────────────────────────────────────────────
st.markdown('<div class="step-label">Step 01</div>', unsafe_allow_html=True)
st.markdown('<div class="step-title">Upload your data</div>', unsafe_allow_html=True)
st.markdown('<div class="step-desc">Three files required — all in Excel (.xlsx) or CSV format.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="upload-card-title">📊 Historical Clients</div>
    <div class="upload-card-desc">Clients with known demand data from existing locations</div>
    """, unsafe_allow_html=True)
    hist_file = st.file_uploader("historical", type=['xlsx','csv'], label_visibility="collapsed")

with col2:
    st.markdown("""
    <div class="upload-card-title">🎯 Potential Clients</div>
    <div class="upload-card-desc">New clients in the target market you want to serve</div>
    """, unsafe_allow_html=True)
    pot_file = st.file_uploader("potential", type=['xlsx','csv'], label_visibility="collapsed")

with col3:
    st.markdown("""
    <div class="upload-card-title">📍 Candidate Sites</div>
    <div class="upload-card-desc">Possible store locations you're considering opening</div>
    """, unsafe_allow_html=True)
    sites_file = st.file_uploader("sites", type=['xlsx','csv'], label_visibility="collapsed")

# ── Load files ─────────────────────────────────────────────────────────────────
def load_file(f):
    if f is None:
        return None
    if f.name.endswith('.csv'):
        return pd.read_csv(f)
    return pd.read_excel(f)

df_hist  = load_file(hist_file)
df_pot   = load_file(pot_file)
df_sites = load_file(sites_file)

# ── Step 2: Column Mapping ─────────────────────────────────────────────────────
if df_hist is not None and df_pot is not None and df_sites is not None:

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25); 
    border-radius:10px; padding:0.8rem 1.2rem; color:#10B981; font-size:0.85rem; font-weight:500;">
    ✅ &nbsp; All 3 files uploaded successfully — ready to configure.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="step-label">Step 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Map your columns</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-desc">Tell NextStep which columns in your files correspond to each field.</div>', unsafe_allow_html=True)

    hist_cols  = df_hist.columns.tolist()
    pot_cols   = df_pot.columns.tolist()
    sites_cols = df_sites.columns.tolist()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="col-label">Historical clients file</div>', unsafe_allow_html=True)
        demand_col   = st.selectbox("Demand column", hist_cols, help="The target variable — weekly or monthly demand")
        distance_col = st.selectbox("Distance to store column", hist_cols, help="Distance from client to nearest existing store")
        lat_col_h    = st.selectbox("Latitude column", hist_cols)
        lon_col_h    = st.selectbox("Longitude column", hist_cols)
        feature_cols = st.multiselect(
            "Client feature columns",
            [c for c in hist_cols if c not in [demand_col, distance_col]],
            default=[c for c in hist_cols if c not in [demand_col, distance_col, lat_col_h, lon_col_h]][:2],
            help="Numerical features describing each client (size, employees, etc.)"
        )

    with col2:
        st.markdown('<div class="col-label">Potential clients & candidate sites</div>', unsafe_allow_html=True)
        lat_col_p  = st.selectbox("Potential clients — latitude", pot_cols)
        lon_col_p  = st.selectbox("Potential clients — longitude", pot_cols)
        lat_col_s  = st.selectbox("Candidate sites — latitude", sites_cols)
        lon_col_s  = st.selectbox("Candidate sites — longitude", sites_cols)

    # ── Step 3: Run ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="step-label">Step 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Run the optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-desc">Our AI will train a demand model and find the mathematically optimal store locations.</div>', unsafe_allow_html=True)

    if st.button("🚀  Find Optimal Locations", type="primary", use_container_width=True):

        with st.spinner("Training AI model and running optimization..."):
            try:
                # Train
                train_features = feature_cols + [distance_col]
                X_train = df_hist[train_features]
                y_train = df_hist[demand_col]
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Distances
                client_coords = df_pot[[lat_col_p, lon_col_p]].values
                site_coords   = df_sites[[lat_col_s, lon_col_s]].values
                dists         = distance_matrix(client_coords, site_coords)
                n_clients     = len(df_pot)
                n_sites       = len(df_sites)

                # Demand matrix
                demand = {}
                for i in range(n_clients):
                    for j in range(n_sites):
                        cf   = df_pot.loc[i, feature_cols].values
                        dist = dists[i, j]
                        xv   = np.append(cf, dist).reshape(1, -1)
                        demand[i, j] = max(0, model.predict(xv)[0])

                # Optimization
                prob = pulp.LpProblem("NextStep", pulp.LpMaximize)
                yv = {j: pulp.LpVariable(f"open_{j}", cat='Binary') for j in range(n_sites)}
                xv2 = {(i,j): pulp.LpVariable(f"assign_{i}_{j}", cat='Binary')
                       for i in range(n_clients) for j in range(n_sites)}

                prob += pulp.lpSum(demand[i,j] * xv2[i,j]
                                   for i in range(n_clients) for j in range(n_sites))
                prob += pulp.lpSum(yv[j] for j in range(n_sites)) <= max_stores

                for i in range(n_clients):
                    for j in range(n_sites):
                        prob += xv2[i,j] <= yv[j]
                for i in range(n_clients):
                    prob += pulp.lpSum(xv2[i,j] for j in range(n_sites)) == 1

                prob.solve(pulp.PULP_CBC_CMD(msg=0))

                opened  = [j for j in range(n_sites) if pulp.value(yv[j]) > 0.5]
                total_d = round(pulp.value(prob.objective), 2)

                rows = []
                for i in range(n_clients):
                    for j in range(n_sites):
                        if pulp.value(xv2[i,j]) > 0.5:
                            rows.append({
                                'client': i,
                                'assigned_site': j,
                                'predicted_demand': round(demand[i,j], 2),
                                'distance': round(dists[i,j], 2),
                                'lat': df_pot.loc[i, lat_col_p],
                                'lon': df_pot.loc[i, lon_col_p]
                            })

                df_asgn = pd.DataFrame(rows)

                # ── Results ───────────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="step-label">Results</div>', unsafe_allow_html=True)
                st.markdown('<div class="step-title">Optimization complete</div>', unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Status", "Optimal ✅")
                m2.metric("Total Weekly Demand", f"{total_d:,.0f}")
                m3.metric("Sites Opened", len(opened))
                m4.metric("Clients Assigned", len(df_asgn))

                st.markdown("<br>", unsafe_allow_html=True)

                # Map
                st.markdown('<div class="section-header">📍 Assignment Map</div>', unsafe_allow_html=True)

                df_asgn['Site'] = "Site " + df_asgn['assigned_site'].astype(str)

                fig = px.scatter(
                    df_asgn,
                    x='lon', y='lat',
                    color='Site',
                    size='predicted_demand',
                    size_max=18,
                    hover_data={'predicted_demand': True, 'distance': True,
                                'lon': False, 'lat': False},
                    labels={'predicted_demand': 'Demand', 'distance': 'Distance'},
                    color_discrete_sequence=['#6366F1','#06B6D4','#10B981',
                                             '#F59E0B','#EF4444','#8B5CF6']
                )

                for j in opened:
                    fig.add_trace(go.Scatter(
                        x=[df_sites.loc[j, lon_col_s]],
                        y=[df_sites.loc[j, lat_col_s]],
                        mode='markers',
                        marker=dict(symbol='star', size=22,
                                    color='#FFFFFF',
                                    line=dict(color='#6366F1', width=2)),
                        name=f'Site {j} ★',
                        showlegend=True
                    ))

                fig.update_layout(
                    paper_bgcolor='#0F0F1A',
                    plot_bgcolor='#0F0F1A',
                    font=dict(family='DM Sans', color='#888899'),
                    legend=dict(bgcolor='#13131F', bordercolor='#1E1E2E',
                                borderwidth=1, font=dict(size=11)),
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=480,
                    xaxis=dict(gridcolor='#1A1A2A', zerolinecolor='#1A1A2A'),
                    yaxis=dict(gridcolor='#1A1A2A', zerolinecolor='#1A1A2A'),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Table
                st.markdown('<div class="section-header">📊 Demand by Site</div>', unsafe_allow_html=True)

                summary = df_asgn.groupby('assigned_site').agg(
                    Clients=('client','count'),
                    Total_Demand=('predicted_demand','sum'),
                    Avg_Demand=('predicted_demand','mean'),
                    Avg_Distance=('distance','mean')
                ).round(1).reset_index()
                summary.columns = ['Site','Clients','Total Demand','Avg Demand / Client','Avg Distance']

                st.dataframe(summary, use_container_width=True, hide_index=True)

                # Download
                st.markdown("<br>", unsafe_allow_html=True)
                csv = df_asgn.to_csv(index=False)
                st.download_button(
                    "⬇️  Download full results as CSV",
                    csv, "nextstep_results.csv", "text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.markdown(f"""
                <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.25);
                border-radius:10px; padding:1rem 1.2rem; color:#EF4444; font-size:0.85rem;">
                ⚠️ &nbsp; Something went wrong: {e}
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(99,102,241,0.06); border:1px dashed #2A2A3E;
    border-radius:12px; padding:2rem; text-align:center; color:#444455;">
        <div style="font-size:2rem; margin-bottom:0.5rem;">📂</div>
        <div style="font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:600; 
        color:#555566; margin-bottom:0.3rem;">Upload all 3 files to get started</div>
        <div style="font-size:0.8rem;">Historical clients · Potential clients · Candidate sites</div>
    </div>
    """, unsafe_allow_html=True)
