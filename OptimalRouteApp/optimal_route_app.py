import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import openrouteservice
from sklearn.cluster import KMeans
from io import BytesIO

# Streamlit setup
st.set_page_config(page_title="Multi Route Optimizer", layout="wide")
st.title("🧭 Optimal Multi-Route Generator")
st.markdown("Upload Excel files with coordinates and cluster them into optimal routes.")

# Set background color
st.markdown("""
<style>
body {
    background-color: #e6f2ff;
}
</style>
""", unsafe_allow_html=True)

# Upload section
uploaded_files = st.file_uploader("📁 Upload up to 10 Excel files", type=["xlsx"], accept_multiple_files=True)
num_routes = st.number_input("🔢 Number of Routes", min_value=1, max_value=10, step=1)
max_capacity = 54  # Max people per route

# OpenRouteService API key
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjFhN2I5N2Q4YmU1MDRjOWZiNmNlNmEzZDA3MjUxMDgxIiwiaCI6Im11cm11cjY0In0="  # Replace with your real ORS key
client = openrouteservice.Client(key=ORS_API_KEY)

@st.cache_data(show_spinner=False)
def generate_clustered_map(df_all, num_routes, max_capacity, _client):
    # Get start point
    start_point = df_all[df_all['Address'] == 'Point 1'].iloc[0]
    start_coord = (start_point['Latitudes'], start_point['Longitudes'])

    clustered_df = df_all[df_all['Address'] != 'Point 1'].copy()

    # KMeans Clustering
    kmeans = KMeans(n_clusters=num_routes, random_state=42)
    coords = clustered_df[['Latitudes', 'Longitudes']]
    clustered_df['Cluster'] = kmeans.fit_predict(coords)

    # Create base map
    m = folium.Map(location=start_coord, zoom_start=12)

    for i in range(num_routes):
        cluster = clustered_df[clustered_df['Cluster'] == i].reset_index(drop=True)

        # Prepare coordinates for routing: start point + cluster points
        route_coords = [(start_coord[1], start_coord[0])] + [
            (row['Longitudes'], row['Latitudes']) for _, row in cluster.iterrows()
        ]

        try:
            route = _client.directions(
                coordinates=route_coords,
                profile='driving-car',
                optimize_waypoints=True,
                format='geojson'
            )
            folium.GeoJson(route, name=f"Route {i+1}").add_to(m)

            # Extract segment distances and compute cumulative distances
            segments = route['features'][0]['properties']['segments']
            cumulative_distances = [0]
            for seg in segments:
                cumulative_distances.append(cumulative_distances[-1] + seg['distance'])

            # Get optimized waypoint order - maps route index to input coords index
            waypoint_order = route['features'][0]['properties']['way_points']

            # Reorder cluster DataFrame according to ORS optimized order (skip start point index 0)
            reordered_indices = waypoint_order[1:]  # drop start point index 0
            reordered_cluster = cluster.iloc[[idx-1 for idx in reordered_indices]].reset_index(drop=True)

            # Now iterate reordered_cluster in visiting order:
            for drop_num, (idx, row) in enumerate(reordered_cluster.iterrows(), start=1):
                cum_dist_km = cumulative_distances[drop_num] / 1000  # meters to km

                popup = f"""
                <b>{row['Address']}</b><br>
                👤 {row['Employee Number']}<br>
                📏 {row['Distance before(km)']} km<br>
                🚗 Drop #{drop_num} — {cum_dist_km:.2f} km cumulative distance
                """
                folium.Marker(
                    location=(row['Latitudes'], row['Longitudes']),
                    tooltip=popup,
                    icon=folium.Icon(color='blue', icon='user')
                ).add_to(m)

        except Exception as e:
            # Fallback: show points without optimized order and cumulative distance
            for _, row in cluster.iterrows():
                popup = f"{row['Address']}<br>{row['Employee Number']}<br>{row['Distance before(km)']}"
                folium.Marker(
                    location=(row['Latitudes'], row['Longitudes']),
                    tooltip=popup,
                    icon=folium.Icon(color='gray', icon='exclamation-triangle', prefix='fa')
                ).add_to(m)

    # Mark start point
    folium.Marker(
        location=start_coord,
        tooltip="Start Point (Point 1)",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)

    return m

# Main logic
if uploaded_files and num_routes:
    df_all = pd.DataFrame()
    error_loading = False
    for file in uploaded_files:
        try:
            df = pd.read_excel(file)
            df_all = pd.concat([df_all, df], ignore_index=True)
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
            error_loading = True
            break

    if not error_loading:
        df_all['Latitudes'] = pd.to_numeric(df_all['Latitudes'], errors='coerce')
        df_all['Longitudes'] = pd.to_numeric(df_all['Longitudes'], errors='coerce')
        df_all.dropna(subset=['Latitudes', 'Longitudes'], inplace=True)

        if 'Point 1' not in df_all['Address'].values:
            st.error("Starting point 'Point 1' not found in data.")
        elif len(df_all) - 1 < num_routes:
            st.error("Not enough points to create that many routes (excluding 'Point 1').")
        elif (len(df_all) - 1) / num_routes > max_capacity:
            min_routes = ((len(df_all) - 1) + max_capacity - 1) // max_capacity
            st.error(f"Too many points for {num_routes} routes. You need at least {min_routes} routes to keep max 54 per route.")
        else:
            m = generate_clustered_map(df_all, num_routes, max_capacity, client)

            st.success(f"✅ Generated {num_routes} optimized route(s).")

            map_html_io = BytesIO()
            m.save(map_html_io, close_file=False)
            map_html_io.seek(0)

            col1, col2 = st.columns([9, 1])
            with col1:
                st_folium(m, width=900, height=600)
            with col2:
                st.download_button(
                    label="📥 Download Map",
                    data=map_html_io,
                    file_name="optimal_routes_map.html",
                    mime="text/html",
                    help="Download the interactive map as an HTML file."
                )
else:
    st.info("Please upload Excel file(s) and select number of routes to generate the map.")
