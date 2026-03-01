import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.ensemble import RandomForestRegressor
import pulp
import plotly.express as px

st.set_page_config(page_title="NextStep", page_icon="📍", layout="wide")

st.title("📍 NextStep")
st.subheader("Find the optimal locations for your business — in minutes.")

st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    max_stores = st.slider("Maximum stores to open", 1, 10, 3)
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload your 3 files")
    st.markdown("2. Map your columns")
    st.markdown("3. Click Find Optimal Locations")
    st.markdown("4. Get your results instantly")

# ── File Upload ────────────────────────────────────────────────────────────────
st.header("Step 1 — Upload your data")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Historical Clients**")
    hist_file = st.file_uploader("Clients with known demand", type=['xlsx','csv'])

with col2:
    st.markdown("**Potential Clients**")
    pot_file = st.file_uploader("New clients to serve", type=['xlsx','csv'])

with col3:
    st.markdown("**Candidate Sites**")
    sites_file = st.file_uploader("Possible store locations", type=['xlsx','csv'])

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

# ── Column Mapping ─────────────────────────────────────────────────────────────
if df_hist is not None and df_pot is not None and df_sites is not None:
    
    st.success("✅ All 3 files uploaded successfully!")
    st.divider()
    st.header("Step 2 — Map your columns")
    st.markdown("Tell NextStep which columns correspond to what.")

    hist_cols  = df_hist.columns.tolist()
    pot_cols   = df_pot.columns.tolist()
    sites_cols = df_sites.columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Historical clients file**")
        demand_col   = st.selectbox("Which column is the DEMAND?", hist_cols)
        distance_col = st.selectbox("Which column is the DISTANCE to store?", hist_cols)
        lat_col_h    = st.selectbox("Which column is LATITUDE?", hist_cols)
        lon_col_h    = st.selectbox("Which column is LONGITUDE?", hist_cols)
        feature_cols = st.multiselect("Which columns are CLIENT FEATURES?", 
                                       [c for c in hist_cols if c not in [demand_col, distance_col]],
                                       default=[c for c in hist_cols if c not in [demand_col, distance_col, lat_col_h, lon_col_h]][:2])

    with col2:
        st.markdown("**Potential clients & sites files**")
        lat_col_p  = st.selectbox("Potential clients — LATITUDE column", pot_cols)
        lon_col_p  = st.selectbox("Potential clients — LONGITUDE column", pot_cols)
        lat_col_s  = st.selectbox("Candidate sites — LATITUDE column", sites_cols)
        lon_col_s  = st.selectbox("Candidate sites — LONGITUDE column", sites_cols)

    st.divider()

    # ── Run ────────────────────────────────────────────────────────────────────
    st.header("Step 3 — Find optimal locations")

    if st.button("🚀 Find Optimal Locations", type="primary", use_container_width=True):

        with st.spinner("Running optimization... this takes about 30 seconds."):

            try:
                # Train model
                train_features = feature_cols + [distance_col]
                X_train = df_hist[train_features]
                y_train = df_hist[demand_col]
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Distances
                client_coords = df_pot[[lat_col_p, lon_col_p]].values
                site_coords   = df_sites[[lat_col_s, lon_col_s]].values
                distances     = distance_matrix(client_coords, site_coords)

                n_clients = len(df_pot)
                n_sites   = len(df_sites)

                # Demand matrix
                demand = {}
                for i in range(n_clients):
                    for j in range(n_sites):
                        client_features = df_pot.loc[i, feature_cols].values
                        dist = distances[i, j]
                        x = np.append(client_features, dist).reshape(1, -1)
                        pred = model.predict(x)[0]
                        demand[i, j] = max(0, pred)

                # Optimization
                prob = pulp.LpProblem("NextStep", pulp.LpMaximize)
                y = {j: pulp.LpVariable(f"open_{j}", cat='Binary') for j in range(n_sites)}
                x = {(i,j): pulp.LpVariable(f"assign_{i}_{j}", cat='Binary')
                     for i in range(n_clients) for j in range(n_sites)}

                prob += pulp.lpSum(demand[i,j] * x[i,j]
                                   for i in range(n_clients)
                                   for j in range(n_sites))
                prob += pulp.lpSum(y[j] for j in range(n_sites)) <= max_stores

                for i in range(n_clients):
                    for j in range(n_sites):
                        prob += x[i,j] <= y[j]

                for i in range(n_clients):
                    prob += pulp.lpSum(x[i,j] for j in range(n_sites)) == 1

                prob.solve(pulp.PULP_CBC_CMD(msg=0))

                # Results
                opened_sites = [j for j in range(n_sites) if pulp.value(y[j]) > 0.5]
                total_demand = round(pulp.value(prob.objective), 2)

                assignments = []
                for i in range(n_clients):
                    for j in range(n_sites):
                        if pulp.value(x[i,j]) > 0.5:
                            assignments.append({
                                'client': i,
                                'assigned_site': j,
                                'predicted_demand': round(demand[i,j], 2),
                                'distance': round(distances[i,j], 2),
                                'lat': df_pot.loc[i, lat_col_p],
                                'lon': df_pot.loc[i, lon_col_p]
                            })

                df_assignments = pd.DataFrame(assignments)

                # ── Display Results ────────────────────────────────────────────
                st.divider()
                st.header("✅ Results")

                m1, m2, m3 = st.columns(3)
                m1.metric("Status", "Optimal ✅")
                m2.metric("Total Weekly Demand", f"{total_demand:,.0f} units")
                m3.metric("Sites Opened", len(opened_sites))

                st.divider()

                # Map
                st.subheader("📍 Map of assignments")

                df_assignments['site_label'] = df_assignments['assigned_site'].astype(str)

                fig = px.scatter(
                    df_assignments,
                    x='lon', y='lat',
                    color='site_label',
                    size='predicted_demand',
                    hover_data=['predicted_demand', 'distance'],
                    title='Client assignments by site',
                    labels={'site_label': 'Assigned Site'},
                    color_discrete_sequence=px.colors.qualitative.Set1
                )

                # Add site markers
                for j in opened_sites:
                    fig.add_scatter(
                        x=[df_sites.loc[j, lon_col_s]],
                        y=[df_sites.loc[j, lat_col_s]],
                        mode='markers',
                        marker=dict(symbol='star', size=20, color='black'),
                        name=f'Site {j} (opened)',
                        showlegend=True
                    )

                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Table
                st.divider()
                st.subheader("📊 Demand by site")

                site_summary = df_assignments.groupby('assigned_site').agg(
                    clients=('client', 'count'),
                    total_demand=('predicted_demand', 'sum'),
                    avg_demand=('predicted_demand', 'mean')
                ).round(2).reset_index()

                site_summary.columns = ['Site', 'Clients Assigned', 
                                        'Total Demand', 'Avg Demand/Client']
                st.dataframe(site_summary, use_container_width=True)

                # Download
                st.divider()
                st.subheader("⬇️ Download results")
                csv = df_assignments.to_csv(index=False)
                st.download_button("Download full assignment table", 
                                   csv, "nextstep_results.csv", "text/csv",
                                   use_container_width=True)

            except Exception as e:
                st.error(f"Something went wrong: {e}")

else:
    st.info("👆 Please upload all 3 files to continue.")