import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import openrouteservice
from sklearn.cluster import KMeans
from io import BytesIO
from math import radians, sin, cos, sqrt, atan2

# =========================
# Streamlit setup
# =========================
st.set_page_config(page_title="Multi Route Optimizer (Open-TSP + Roadside)", layout="wide")
st.title("🧭 Optimal Multi-Route Generator (Open Route, Exact Road Distance)")
st.markdown("Upload Excel files with coordinates. The app clusters points, computes **road-distance optimal** drop orders (open route), and shows **cumulative driving distance** from the depot. Supports roadside drop (snap to driven route).")

# Light background
st.markdown("""
<style>
body { background-color: #e6f2ff; }
</style>
""", unsafe_allow_html=True)

# =========================
# Inputs
# =========================
uploaded_files = st.file_uploader("📁 Upload up to 10 Excel files", type=["xlsx"], accept_multiple_files=True)
num_routes = st.number_input("🔢 Number of Routes", min_value=1, max_value=10, step=1)
max_capacity_business = 65  # Your business capacity per route
roadside_snap = st.toggle("🚏 Roadside drop (snap to nearest point on driven route)", value=True)
st.caption("Roadside drop avoids unnecessary u-turns just to touch the exact GPS pin across a divided road.")

# =========================
# OpenRouteService API key (constant)
# =========================
# 🔐 Replace with your actual ORS API key (keep it constant here as requested)
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjFhN2I5N2Q4YmU1MDRjOWZiNmNlNmEzZDA3MjUxMDgxIiwiaCI6Im11cm11cjY0In0="
if not ORS_API_KEY or ORS_API_KEY.strip() == "":
    st.error("OpenRouteService API key is missing in the script.")
    st.stop()

client = openrouteservice.Client(key=ORS_API_KEY)

# =========================
# Helpers (geometry & routing)
# =========================
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlmb = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1) * cos(phi2) * sin(dlmb/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def build_matrix(coords_lonlat, _client):
    """
    Build a road-distance matrix (meters) using ORS /matrix (may have plan limits ~50 points total).
    coords_lonlat: list of (lon, lat)
    """
    resp = _client.distance_matrix(
        locations=coords_lonlat,
        profile='driving-car',
        metrics=['distance'],
        resolve_locations=True
    )
    return np.array(resp['distances'], dtype=float)

def nearest_neighbor_open(dist_mat, start_idx=0):
    n = dist_mat.shape[0]
    unvisited = set(range(n)); unvisited.remove(start_idx)
    order = [start_idx]; cur = start_idx
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist_mat[cur, j])
        order.append(nxt); unvisited.remove(nxt); cur = nxt
    return order

def two_opt_open(order, dist_mat, max_iters=2000):
    def route_len(ordr):
        return sum(dist_mat[ordr[i], ordr[i+1]] for i in range(len(ordr)-1))
    best = order[:]; best_len = route_len(best)
    improved = True; iters = 0
    while improved and iters < max_iters:
        improved = False; iters += 1
        for i in range(1, len(best)-2):
            for k in range(i+1, len(best)-1):
                cand = best[:i] + best[i:k+1][::-1] + best[k+1:]
                cand_len = sum(dist_mat[cand[t], cand[t+1]] for t in range(len(cand)-1))
                if cand_len + 1e-6 < best_len:
                    best, best_len = cand, cand_len
                    improved = True
                    break
            if improved: break
    return best, best_len

def compute_open_tsp_order(coords_lonlat, use_matrix=True):
    """
    Returns an open-route order and length.
    - If use_matrix: use ORS matrix -> NN -> 2-opt
    - Else: use Haversine distances for ordering -> 2-opt on haversine
    """
    n = len(coords_lonlat)
    if use_matrix:
        mat = build_matrix(coords_lonlat, client)
    else:
        # Haversine matrix (approx) to avoid ORS matrix limits for huge clusters
        mat = np.zeros((n, n), dtype=float)
        latlon = [(c[1], c[0]) for c in coords_lonlat]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                mat[i, j] = haversine_m(latlon[i][0], latlon[i][1], latlon[j][0], latlon[j][1])

    nn_order = nearest_neighbor_open(mat, start_idx=0)
    best_order, best_len = two_opt_open(nn_order, mat)
    return best_order, best_len, mat

def fetch_directions_pair(a_lonlat, b_lonlat, _client):
    """
    Get driving directions for a->b.
    Returns (feature, meters, line_lonlat)
    """
    route = _client.directions(
        coordinates=[a_lonlat, b_lonlat],
        profile="driving-car",
        format="geojson"
    )
    feat = route['features'][0]
    seg = feat['properties']['segments'][0]
    line = feat['geometry']['coordinates']  # [ [lon,lat], ... ]
    return feat, seg['distance'], line

def concat_route_features(order, coords_lonlat, _client):
    """
    Returns:
      features: list of GeoJSON features per leg
      leg_distances_m: list[meters]
      poly_latlon: concatenated driven polyline [(lat,lon), ...]
      waypoint_cum_m: cumulative meters at each way-point in 'order' (start @ 0)
    """
    features = []
    leg_distances = []
    poly_latlon = []
    cumulative = [0.0]
    prev_end = None

    for i in range(len(order) - 1):
        a = coords_lonlat[order[i]]
        b = coords_lonlat[order[i+1]]
        feat, d_m, line_lonlat = fetch_directions_pair(a, b, _client)
        features.append(feat)
        leg_distances.append(d_m)

        line_latlon = [(pt[1], pt[0]) for pt in line_lonlat]
        if not poly_latlon:
            poly_latlon.extend(line_latlon)
        else:
            # Avoid duplicating the stitch vertex
            if prev_end == line_latlon[0]:
                poly_latlon.extend(line_latlon[1:])
            else:
                poly_latlon.extend(line_latlon)
        prev_end = line_latlon[-1]

        cumulative.append(cumulative[-1] + d_m)

    return features, leg_distances, poly_latlon, cumulative

def cumulative_along_polyline(poly_latlon):
    """Cumulative meters at each polyline vertex."""
    cum = [0.0]
    for i in range(1, len(poly_latlon)):
        d = haversine_m(poly_latlon[i-1][0], poly_latlon[i-1][1], poly_latlon[i][0], poly_latlon[i][1])
        cum.append(cum[-1] + d)
    return cum

def project_point_to_segment(p, a, b):
    """
    Project point p=(lat,lon) to segment a->b (lat/lon) using local planar approx.
    Returns (proj_latlon, t in [0,1]).
    """
    lat_scale = 111320.0
    lon_scale = 111320.0 * cos(radians(a[0]))

    ax, ay = 0.0, 0.0
    bx = (b[1] - a[1]) * lon_scale
    by = (b[0] - a[0]) * lat_scale
    px = (p[1] - a[1]) * lon_scale
    py = (p[0] - a[0]) * lat_scale

    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx*abx + aby*aby
    t = 0.0 if ab2 == 0 else max(0.0, min(1.0, (apx*abx + apy*aby) / ab2))

    projx = ax + t*abx
    projy = ay + t*aby
    proj_lon = a[1] + projx / lon_scale
    proj_lat = a[0] + projy / lat_scale
    return (proj_lat, proj_lon), t

def roadside_cumulative_distance(point_latlon, poly_latlon, poly_cum_m):
    """
    Find nearest point on polyline to point_latlon.
    Returns (cumulative_m_at_projection, perpendicular_distance_m)
    """
    nearest_cum = None
    min_dist = float('inf')
    for i in range(len(poly_latlon) - 1):
        a = poly_latlon[i]
        b = poly_latlon[i+1]
        proj, t = project_point_to_segment(point_latlon, a, b)
        d_perp = haversine_m(point_latlon[0], point_latlon[1], proj[0], proj[1])
        if d_perp < min_dist:
            min_dist = d_perp
            seg_len = poly_cum_m[i+1] - poly_cum_m[i]
            nearest_cum = poly_cum_m[i] + t * seg_len
    return nearest_cum, min_dist

def kmeans_cluster(df, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    out = df.copy()
    out['Cluster'] = kmeans.fit_predict(out[['Latitudes', 'Longitudes']])
    return out

def fmt_km(m): return f"{m/1000:.2f} km"

# =========================
# Main workflow
# =========================
if uploaded_files and num_routes:
    # Load & merge
    df_all = pd.DataFrame()
    error_loading = False
    for file in uploaded_files:
        try:
            df = pd.read_excel(file)
            df_all = pd.concat([df_all, df], ignore_index=True)
        except Exception as e:
            st.error(f"Error reading {getattr(file, 'name', 'file')}: {e}")
            error_loading = True
            break

    if not error_loading:
        # Ensure numeric types
        for col in ['Latitudes', 'Longitudes']:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
        df_all.dropna(subset=['Latitudes', 'Longitudes'], inplace=True)

        if 'Address' not in df_all.columns:
            st.error("Column 'Address' is required.")
        elif 'Point 1' not in df_all['Address'].values:
            st.error("Starting point 'Point 1' not found in data.")
        elif len(df_all) - 1 < num_routes:
            st.error("Not enough points to create that many routes (excluding 'Point 1').")
        else:
            # Depot
            start_row = df_all[df_all['Address'] == 'Point 1'].iloc[0]
            start_coord_latlon = (float(start_row['Latitudes']), float(start_row['Longitudes']))

            # Exclude depot for clustering
            drops_df = df_all[df_all['Address'] != 'Point 1'].copy()

            # Business capacity check
            if (len(drops_df) / num_routes) > max_capacity_business:
                min_routes = int(np.ceil(len(drops_df) / max_capacity_business))
                st.error(f"Too many points for {num_routes} routes. You need at least {min_routes} routes to keep max {max_capacity_business} per route.")
            else:
                # Cluster
                clustered_df = kmeans_cluster(drops_df, num_routes)

                # Create base map
                m = folium.Map(location=start_coord_latlon, zoom_start=12, control_scale=True)

                total_all_routes = 0.0
                colors = ["blue","green","purple","orange","darkred","cadetblue","darkpurple","darkgreen","pink","lightblue"]

                # Plot each route
                for i in range(num_routes):
                    cluster = clustered_df[clustered_df['Cluster'] == i].copy()
                    if cluster.empty:
                        continue

                    # Coordinates for this route: depot + cluster points  (ORS expects (lon, lat))
                    coords_lonlat = [(start_coord_latlon[1], start_coord_latlon[0])]
                    cluster_points = list(zip(cluster['Longitudes'].astype(float), cluster['Latitudes'].astype(float)))
                    coords_lonlat.extend(cluster_points)

                    # Decide whether we can use ORS matrix (safer up to ~50 nodes total)
                    use_matrix = len(coords_lonlat) <= 50

                    # Compute open-route order (start fixed at index 0)
                    try:
                        order, _, _ = compute_open_tsp_order(coords_lonlat, use_matrix=use_matrix)
                    except Exception as e:
                        st.error(f"Routing error in Route {i+1} while computing order: {e}")
                        continue

                    # Fetch per-leg directions and concatenated polyline
                    try:
                        features, leg_dists, poly_latlon, wp_cum = concat_route_features(order, coords_lonlat, client)
                    except Exception as e:
                        st.error(f"Directions fetch error in Route {i+1}: {e}")
                        continue

                    # Polyline cumulative (for roadside snapping)
                    poly_cum = cumulative_along_polyline(poly_latlon)

                    # Draw route segments
                    route_color = colors[i % len(colors)]
                    fg = folium.FeatureGroup(name=f"Route {i+1}")
                    for feat in features:
                        folium.GeoJson(
                            data=feat,
                            name=f"Route {i+1}",
                            style_function=lambda x, c=route_color: {"color": c, "weight": 4, "opacity": 0.9}
                        ).add_to(fg)
                    fg.add_to(m)

                    # Mark start point
                    folium.Marker(
                        location=start_coord_latlon,
                        tooltip=f"Start (Depot: Point 1) | Route {i+1}",
                        icon=folium.Icon(color="red", icon="home")
                    ).add_to(m)

                    # Identify last drop (optimal end)
                    last_idx = order[-1]
                    last_is_depot = (last_idx == 0)

                    # Add markers with cumulative distance popup
                    ordered_labels = []
                    total_route_m = wp_cum[-1]
                    total_all_routes += total_route_m

                    for pos, idx in enumerate(order):
                        if idx == 0:
                            ordered_labels.append("Depot (Point 1)")
                            continue

                        row = cluster.iloc[idx - 1]
                        lat, lon = float(row['Latitudes']), float(row['Longitudes'])
                        address = str(row.get('Address', 'N/A'))
                        emp = str(row.get('Employee Number', 'N/A'))
                        before_km = row.get('Distance before(km)', None)

                        # Default cumulative at waypoint boundary (sum of driven legs up to this stop)
                        cum_waypoint_m = wp_cum[pos]

                        # Roadside snap to driven route
                        popup_lines = [f"<b>{address}</b>", f"👤 {emp}"]
                        if roadside_snap and len(poly_latlon) >= 2:
                            cum_roadside_m, d_perp_m = roadside_cumulative_distance((lat, lon), poly_latlon, poly_cum)
                            if cum_roadside_m is not None:
                                # Keep non-decreasing vs previous
                                if pos > 0:
                                    cum_roadside_m = max(cum_roadside_m, wp_cum[pos-1])
                                popup_lines.append(f"🚗 Cumulative from depot (roadside): {cum_roadside_m/1000:.2f} km")
                                popup_lines.append(f"↔︎ Offset to road: {d_perp_m:.0f} m")
                            else:
                                popup_lines.append(f"🚗 Cumulative from depot: {cum_waypoint_m/1000:.2f} km")
                        else:
                            popup_lines.append(f"🚗 Cumulative from depot: {cum_waypoint_m/1000:.2f} km")

                        if pd.notna(before_km):
                            popup_lines.append(f"📏 Previous (provided): {before_km} km")

                        icon_color = "blue" if idx != last_idx else "darkred"
                        folium.Marker(
                            location=(lat, lon),
                            tooltip="<br>".join(popup_lines),
                            icon=folium.Icon(color=icon_color, icon='user')
                        ).add_to(m)

                        label = f"{address} (Emp: {emp})"
                        if idx == last_idx:
                            label += "  ⟵ last drop"
                        ordered_labels.append(label)

                    # Route summary
                    st.markdown(f"### 🗺️ Route {i+1} Summary")
                    st.write(
                        f"- **Stops (excl. depot):** {len(cluster)}\n"
                        f"- **Total driving distance:** {fmt_km(total_route_m)}\n"
                        f"- **Last drop (optimal end):** {'Depot (check data)' if last_is_depot else 'Employee drop'}"
                        
                    )
                    st.write("**Drop Order (open route):**")
                    st.write(" ➜ ".join(ordered_labels))

                # Layer control
                folium.LayerControl().add_to(m)

                # Success + Map + Download
                st.success(f"✅ Generated {num_routes} open, optimized route(s). **Sum of route distances:** {fmt_km(total_all_routes)}")
                col1, col2 = st.columns([9, 1])
                with col1:
                    st_folium(m, width=1000, height=650)
                with col2:
                    map_html_io = BytesIO()
                    m.save(map_html_io, close_file=False)
                    map_html_io.seek(0)
                    st.download_button(
                        label="📥 Download Map",
                        data=map_html_io,
                        file_name="optimal_routes_open_tsp_roadside.html",
                        mime="text/html",
                        help="Download the interactive map as an HTML file."
                    )
else:
    st.info("Please upload Excel file(s) and select number of routes to generate the map.")
