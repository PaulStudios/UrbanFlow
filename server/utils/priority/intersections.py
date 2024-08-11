import logging
import warnings

import folium
import googlemaps
import osmnx as ox
from polyline import decode
from sklearn.cluster import DBSCAN
import numpy as np

# Google Maps API key
API_KEY = 'AIzaSyD2eZigNOxSBMtC66lC4Tu-xCg-_F4NASI'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


gmaps = googlemaps.Client(key=API_KEY)


def get_route(origin, destination):
    """Get the route between origin and destination."""
    logger.info(f"Fetching route from {origin} to {destination}")
    directions_result = gmaps.directions(origin, destination, mode="driving")
    polyline = directions_result[0]['overview_polyline']['points']
    route_points = decode(polyline)
    logger.info(f"Route fetched with {len(route_points)} points.")
    return route_points


def fetch_osm_intersections(lat, lon, dist=300):
    """Fetch intersections from OpenStreetMap using osmnx near the given latitude and longitude."""
    try:
        G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive')
        if G.number_of_nodes() == 0:
            logger.warning(
                f"No graph nodes found within the requested polygon for coordinates ({lat}, {lon}). Skipping...")
            return []

        nodes, streets = ox.graph_to_gdfs(G, nodes=True, edges=True)
        intersections = nodes[nodes['street_count'] > 1]
        return intersections[['y', 'x']].values.tolist()
    except Exception as e:
        logger.error(f"Error fetching OSM data for coordinates ({lat}, {lon}): {e}")
        return []


def find_intersections(route_points, radius=50):
    """Find intersections along the route."""
    intersections = []
    logger.info("Finding intersections along the route.")
    for lat, lon in route_points:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            osm_intersections = fetch_osm_intersections(lat, lon, radius)
        intersections.extend(osm_intersections)
    logger.info(f"Found {len(intersections)} intersections along the route.")
    return intersections


def cluster_intersections(intersections, eps=0.00001, min_samples=1):
    """Cluster intersections using DBSCAN to remove duplicates."""
    if not intersections:
        return []

    logger.info("Clustering intersections using DBSCAN.")
    coords = np.array(intersections)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine').fit(np.radians(coords))

    # Extract the cluster centers
    cluster_labels = db.labels_
    unique_labels = set(cluster_labels)

    clustered_intersections = []
    for k in unique_labels:
        if k != -1:  # Ignore noise points
            class_member_mask = (cluster_labels == k)
            cluster_coords = coords[class_member_mask]
            cluster_center = cluster_coords.mean(axis=0)
            clustered_intersections.append(tuple(cluster_center))

    logger.info(f"Reduced to {len(clustered_intersections)} clustered intersections.")
    return clustered_intersections


def visualize_route_with_intersections(route_points, intersections, origin, destination):
    """Visualize the route and intersections on a map."""
    logger.info("Visualizing route with intersections on the map.")
    map_center = route_points[len(route_points) // 2]
    route_map = folium.Map(location=map_center, zoom_start=13)

    # Add route to map
    folium.PolyLine(route_points, color="blue", weight=2.5, opacity=1).add_to(route_map)

    # Add intersections to map
    for lat, lon in intersections:
        folium.Marker(location=(lat, lon), icon=folium.Icon(color='red')).add_to(route_map)

    # Add origin and destination markers
    folium.Marker(location=route_points[0], popup=origin, icon=folium.Icon(color='green')).add_to(route_map)
    folium.Marker(location=route_points[-1], popup=destination, icon=folium.Icon(color='green')).add_to(route_map)

    # Save map to HTML file
    map_filename = "route_with_intersections.html"
    route_map.save(map_filename)
    logger.info(f"Map saved to {map_filename}")


def main():
    origin = "Times Square, New York, NY"
    destination = "Central Park, New York, NY"

    origin = "Madhyamgram, Kolkata, IN"
    destination = "Barasat, Kolkata, IN"

    # Fetch and decode the route
    route_points = get_route(origin, destination)

    # Find intersections along the route
    intersections = find_intersections(route_points)

    # Cluster intersections using DBSCAN to remove duplicates
    clustered_intersections = cluster_intersections(intersections)

    print(clustered_intersections)

    # Visualize the route with intersections
    visualize_route_with_intersections(route_points, clustered_intersections, origin, destination)


if __name__ == "__main__":
    main()