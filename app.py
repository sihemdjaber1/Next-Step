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
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #FAFAF8;
    color: #1A1A1A;
}

.stApp { background: #FAFAF8; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stSidebar"] { display: none; }

.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 0;
    border-bottom: 1px solid #E8E8E4;
}

.nav-logo {
    font-family: 'Instrument Serif', serif;
    font-size: 1.5rem;
    color: #1A1A1A;
}

.nav-logo span { color: #2563EB; }
.nav-links { font-size: 0.82rem; color: #AAA; }

.hero-section { padding: 5rem 0 4rem 0; max-width: 720px; }

.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2563EB;
    margin-bottom: 1.2rem;
}

.hero-headline {
    font-family: 'Instrument Serif', serif;
    font-size: 3.8rem;
    line-height: 1.08;
    letter-spacing: -0.02em;
    color: #0F0F0F;
    margin: 0 0 1.4rem 0;
    font-weight: 400;
}

.hero-headline em { font-style: italic; color: #2563EB; }

.hero-body {
    font-size: 1.05rem;
    font-weight: 300;
    color: #555;
    line-height: 1.7;
    max-width: 560px;
    margin-bottom: 2.5rem;
}

.hero-stats {
    display: flex;
    gap: 3rem;
    padding-top: 2rem;
    border-top: 1px solid #E8E8E4;
}

.stat-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #AAA;
    margin-bottom: 0.3rem;
}

.stat-value {
    font-family: 'Instrument Serif', serif;
    font-size: 2rem;
    color: #0F0F0F;
    line-height: 1;
}

.section-eyebrow {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2563EB;
    margin-bottom: 0.5rem;
}

.section-title {
    font-family: 'Instrument Serif', serif;
    font-size: 2rem;
    font-weight: 400;
    color: #0F0F0F;
    letter-spacing: -0.015em;
    margin-bottom: 0.4rem;
    line-height: 1.2;
}

.section-body {
    font-size: 0.88rem;
    color: #777;
    font-weight: 300;
    line-height: 1.6;
    margin-bottom: 1.8rem;
}

.upload-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #1A1A1A;
    margin-bottom: 0.15rem;
}

.upload-sublabel {
    font-size: 0.72rem;
    color: #AAA;
    margin-bottom: 0.6rem;
    font-weight: 300;
}

[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1.5px dashed #D8D8D4 !important;
    border-radius: 8px !important;
}

.success-banner {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 0.85rem 1.2rem;
    color: #1D4ED8;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 1.5rem 0;
}

[data-testid="stSelectbox"] > div > div {
    background: #FFFFFF !important;
    border: 1px solid #E0E0DC !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
}

[data-testid="stMultiSelect"] > div > div {
    background: #FFFFFF !important;
    border: 1px solid #E0E0DC !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
}

.stButton > button[kind="primary"] {
    background: #1A1A1A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    height: 3rem !important;
    transition: all 0.15s !important;
}

.stButton > button[kind="primary"]:hover {
    background: #2563EB !important;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E8E8E4;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
}

[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    color: #AAA !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Instrument Serif', serif !important;
    font-size: 1.9rem !important;
    font-weight: 400 !important;
    color: #0F0F0F !important;
}

hr {
    border: none !important;
    border-top: 1px solid #E8E8E4 !important;
    margin: 2.5rem 0 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #E8E8E4 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: white !important;
}

.stDownloadButton > button {
    background: #FFFFFF !important;
    color: #1A1A1A !important;
    border: 1px solid #E0E0DC !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
}

.col-header {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #AAA;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid #E8E8E4;
    margin-bottom: 1rem;
}

.map-title {
    font-family: 'Instrument Serif', serif;
    font-size: 1.3rem;
    font-weight: 400;
    color: #0F0F0F;
    margin: 1.5rem 0 0.8rem 0;
}

.empty-state {
    background: #FFFFFF;
    border: 1.5px dashed #D8D8D4;
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
    margin-top: 1rem;
}

.empty-icon { font-size: 2rem; margin-bottom: 0.8rem; }

.empty-title {
    font-family: 'Instrument Serif', serif;
    font-size: 1.2rem;
    color: #555;
    margin-bottom: 0.3rem;
}

.empty-sub { font-size: 0.78rem; color: #BBB; font-weight: 300; }

label[data-testid="stWidgetLabel"] p {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #555 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

</style>
""", unsafe_allow_html=True)

# ── Navbar ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="nav-logo">Next<span>Step</span></div>
    <div class="nav-links">Retail Location Intelligence &nbsp;·&nbsp; AI-Powered</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-eyebrow">Location Intelligence Platform</div>
    <h1 class="hero-headline">Open the right stores.<br><em>Every time.</em></h1>
    <p class="hero-body">
        Stop paying €50,000 for location consultants. Upload your data and let our AI 
        find the mathematically optimal store locations to maximize your weekly revenue — 
        in under 5 minutes.
    </p>
    <div class="hero-stats">
        <div>
            <div class="stat-label">vs. Consulting</div>
            <div class="stat-value">100×</div>
        </div>
        <div>
            <div class="stat-label">Time to result</div>
            <div class="stat-value">&lt; 5 min</div>
        </div>
        <div>
            <div class="stat-label">Technology</div>
            <div class="stat-value">AI + Math</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Settings ───────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1,1,2])
with c1:
    max_stores = st.slider("Maximum stores to open", 1, 10, 3)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Step 1 ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-eyebrow">Step 01</div>
<div class="section-title">Upload your data</div>
<div class="section-body">Three files required — Excel (.xlsx) or CSV format.</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="upload-label">Historical Clients</div><div class="upload-sublabel">Existing clients with known demand</div>', unsafe_allow_html=True)
    hist_file = st.file_uploader("hist", type=['xlsx','csv'], label_visibility="collapsed")

with col2:
    st.markdown('<div class="upload-label">Potential Clients</div><div class="upload-sublabel">New clients in your target market</div>', unsafe_allow_html=True)
    pot_file = st.file_uploader("pot", type=['xlsx','csv'], label_visibility="collapsed")

with col3:
    st.markdown('<div class="upload-label">Candidate Sites</div><div class="upload-sublabel">Locations you are considering</div>', unsafe_allow_html=True)
    sites_file = st.file_uploader("sites", type=['xlsx','csv'], label_visibility="collapsed")

def load_file(f):
    if f is None: return None
    return pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)

df_hist  = load_file(hist_file)
df_pot   = load_file(pot_file)
df_sites = load_file(sites_file)

if df_hist is not None and df_pot is not None and df_sites is not None:

    st.markdown('<div class="success-banner">✓ &nbsp; All 3 files loaded — proceed to column mapping below.</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-eyebrow">Step 02</div>
    <div class="section-title">Map your columns</div>
    <div class="section-body">Tell NextStep which columns correspond to each required field.</div>
    """, unsafe_allow_html=True)

    hist_cols  = df_hist.columns.tolist()
    pot_cols   = df_pot.columns.tolist()
    sites_cols = df_sites.columns.tolist()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="col-header">Historical clients file</div>', unsafe_allow_html=True)
        demand_col   = st.selectbox("Demand column", hist_cols)
        distance_col = st.selectbox("Distance to store column", hist_cols)
        lat_col_h    = st.selectbox("Latitude column", hist_cols)
        lon_col_h    = st.selectbox("Longitude column", hist_cols)
        feature_cols = st.multiselect(
            "Client feature columns",
            [c for c in hist_cols if c not in [demand_col, distance_col]],
            default=[c for c in hist_cols if c not in [demand_col, distance_col, lat_col_h, lon_col_h]][:2]
        )

    with col2:
        st.markdown('<div class="col-header">Potential clients & candidate sites</div>', unsafe_allow_html=True)
        lat_col_p = st.selectbox("Potential clients — latitude", pot_cols)
        lon_col_p = st.selectbox("Potential clients — longitude", pot_cols)
        lat_col_s = st.selectbox("Candidate sites — latitude", sites_cols)
        lon_col_s = st.selectbox("Candidate sites — longitude", sites_cols)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-eyebrow">Step 03</div>
    <div class="section-title">Run the optimization</div>
    <div class="section-body">Our AI trains on your historical data and finds the optimal site locations.</div>
    """, unsafe_allow_html=True)

    if st.button("Find Optimal Locations →", type="primary"):

        with st.spinner("Running AI optimization..."):
            try:
                train_features = feature_cols + [distance_col]
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(df_hist[train_features], df_hist[demand_col])

                dists     = distance_matrix(
                    df_pot[[lat_col_p, lon_col_p]].values,
                    df_sites[[lat_col_s, lon_col_s]].values
                )
                n_clients = len(df_pot)
                n_sites   = len(df_sites)

                demand = {}
                for i in range(n_clients):
                    for j in range(n_sites):
                        cf = df_pot.loc[i, feature_cols].values
                        xv = np.append(cf, dists[i,j]).reshape(1,-1)
                        demand[i,j] = max(0, model.predict(xv)[0])

                prob = pulp.LpProblem("NextStep", pulp.LpMaximize)
                yv   = {j: pulp.LpVariable(f"o_{j}", cat='Binary') for j in range(n_sites)}
                xv2  = {(i,j): pulp.LpVariable(f"a_{i}_{j}", cat='Binary')
                        for i in range(n_clients) for j in range(n_sites)}

                prob += pulp.lpSum(demand[i,j]*xv2[i,j]
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

                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("""
                <div class="section-eyebrow">Results</div>
                <div class="section-title">Optimization complete</div>
                """, unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Status", "Optimal")
                m2.metric("Total Weekly Demand", f"{total_d:,.0f}")
                m3.metric("Sites Opened", len(opened))
                m4.metric("Clients Served", len(df_asgn))

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="map-title">Assignment Map</div>', unsafe_allow_html=True)

                df_asgn['Site'] = "Site " + df_asgn['assigned_site'].astype(str)

                fig = px.scatter(
                    df_asgn, x='lon', y='lat',
                    color='Site', size='predicted_demand', size_max=16,
                    hover_data={'predicted_demand': True, 'distance': True,
                                'lon': False, 'lat': False},
                    color_discrete_sequence=['#2563EB','#16A34A','#DC2626',
                                             '#D97706','#7C3AED','#0891B2']
                )

                for j in opened:
                    fig.add_trace(go.Scatter(
                        x=[df_sites.loc[j, lon_col_s]],
                        y=[df_sites.loc[j, lat_col_s]],
                        mode='markers',
                        marker=dict(symbol='star', size=20, color='#1A1A1A',
                                    line=dict(color='white', width=1.5)),
                        name=f'Site {j} (opened)',
                        showlegend=True
                    ))

                fig.update_layout(
                    paper_bgcolor='#FFFFFF', plot_bgcolor='#F8F8F6',
                    font=dict(family='Inter', color='#555', size=11),
                    legend=dict(bgcolor='#FFFFFF', bordercolor='#E8E8E4', borderwidth=1),
                    margin=dict(l=20, r=20, t=20, b=20), height=460,
                    xaxis=dict(gridcolor='#EEEEEA', zerolinecolor='#EEEEEA', title='Longitude'),
                    yaxis=dict(gridcolor='#EEEEEA', zerolinecolor='#EEEEEA', title='Latitude'),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown('<div class="map-title">Demand by Site</div>', unsafe_allow_html=True)
                summary = df_asgn.groupby('assigned_site').agg(
                    Clients=('client','count'),
                    Total_Demand=('predicted_demand','sum'),
                    Avg_Demand=('predicted_demand','mean'),
                    Avg_Distance=('distance','mean')
                ).round(1).reset_index()
                summary.columns = ['Site','Clients','Total Demand','Avg Demand / Client','Avg Distance']
                st.dataframe(summary, use_container_width=True, hide_index=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button("Download results as CSV →",
                                   df_asgn.to_csv(index=False),
                                   "nextstep_results.csv", "text/csv")

            except Exception as e:
                st.markdown(f'<div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;padding:1rem;color:#DC2626;font-size:0.82rem;">⚠ {e}</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📂</div>
        <div class="empty-title">Upload all 3 files to get started</div>
        <div class="empty-sub">Historical clients · Potential clients · Candidate sites</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;justify-content:space-between;padding-bottom:2rem;
font-size:0.72rem;color:#CCC;">
    <div>© 2025 NextStep</div>
    <div>Random Forest ML + Binary Integer Programming</div>
</div>
""", unsafe_allow_html=True)
