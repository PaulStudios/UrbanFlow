import asyncio
import logging
import os
import warnings
from typing import List, Tuple

import folium
import googlemaps
import osmnx as ox
from fastapi_cache.decorator import cache
from polyline import decode
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.neighbors import BallTree

# Google Maps API key
API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

gmaps = googlemaps.Client(key=API_KEY)


def get_route(origin, destination):
    """Get the route between origin and destination with increased accuracy."""
    logger.info(f"Fetching detailed route from {origin} to {destination}")
    directions_result = gmaps.directions(
        origin,
        destination,
        mode="driving",
        alternatives=False,
        waypoints=None,
        optimize_waypoints=False,
        avoid=None,
        language=None,
        units=None,
        region=None,
        departure_time=None,
        arrival_time=None,
        transit_mode=None,
        transit_routing_preference=None,
        traffic_model=None
    )

    # Extract all steps from the route
    steps = directions_result[0]['legs'][0]['steps']

    # Decode polyline for each step and combine
    route_points = []
    for step in steps:
        points = decode(step['polyline']['points'])
        route_points.extend(points)

    logger.info(f"Detailed route fetched with {len(route_points)} points.")
    return route_points


# @cache(expire=3600)
async def fetch_osm_intersections(lat: float, lon: float, dist: int = 1000) -> List[Tuple[float, float]]:
    """Fetch intersections from OpenStreetMap using osmnx near the given latitude and longitude."""
    try:
        with warnings.catch_warnings(action="ignore"):
            G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive')
        if G.number_of_nodes() == 0:
            logger.warning(
                f"No graph nodes found within the requested polygon for coordinates ({lat}, {lon}). Skipping...")
            return []

        nodes, streets = ox.graph_to_gdfs(G, nodes=True, edges=True)
        intersections = nodes[nodes['street_count'] > 1]
        logger.info(f"Found intersection: {intersections[['y','x']].values.tolist()}")
        return intersections[['y', 'x']].values.tolist()
    except Exception as e:
        logger.error(f"Error fetching OSM data for coordinates ({lat}, {lon}): {e}")
        return []


def sample_route_points(route_points, sample_distance=1000):
    """Sample route points at a given distance interval."""
    sampled_points = [route_points[0]]
    cumulative_distance = 0
    for point in route_points[1:]:
        distance = ox.distance.great_circle(sampled_points[-1][0], sampled_points[-1][1], point[0], point[1])
        cumulative_distance += distance
        if cumulative_distance >= sample_distance:
            sampled_points.append(point)
            cumulative_distance = 0
    if sampled_points[-1] != route_points[-1]:
        sampled_points.append(route_points[-1])
    return sampled_points


async def find_intersections(route_points, radius=700):
    """Find intersections along the route."""
    intersections = []
    logger.info("Finding intersections along the route.")

    sampled_points = sample_route_points(route_points)
    logger.info(f"Sampled {len(sampled_points)} points from {len(route_points)} original points.")

    # Use asyncio.gather to run all fetch_osm_intersections calls concurrently
    intersection_lists = await asyncio.gather(*[
        fetch_osm_intersections(lat, lon, radius)
        for lat, lon in sampled_points
    ])

    # Flatten the list of lists into a single list of intersections
    intersections = [point for sublist in intersection_lists for point in sublist]

    logger.info(f"Found {len(intersections)} intersections along the route.")
    return intersections


def cluster_intersections(intersections, eps=0.00001, min_samples=1):
    """Cluster intersections using DBSCAN to remove duplicates."""
    if not intersections:
        return []

    logger.info("Clustering intersections using DBSCAN.")

    # Convert intersections to a numpy array
    coords = np.array(intersections, dtype=float)

    # Apply radians conversion
    coords_rad = np.radians(coords)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine').fit(coords_rad)

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


def filter_intersections(route, intersections, threshold=50):
    # Convert route and intersections to numpy arrays
    route = np.array(route)
    intersections = np.array(intersections)

    # Create a BallTree from the route points
    tree = BallTree(np.radians(route), metric='haversine')

    # Query the tree for each intersection point
    distances, _ = tree.query(np.radians(intersections), k=1)

    # Convert distances from radians to meters (Earth's radius ≈ 6371 km)
    distances = distances.flatten() * 6371000

    # Filter intersections based on the threshold
    mask = distances <= threshold
    filtered_intersections = intersections[mask]

    intersection_list = filtered_intersections.tolist()
    logger.info(f"Reduced to {len(intersection_list)} intersections.")
    return filtered_intersections.tolist()


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
    map_filename = "../route_with_intersections.html"
    route_map.save(map_filename)
    logger.info(f"Map saved to {map_filename}")


async def main():
    origin = "Times Square, New York, NY"
    destination = "Central Park, New York, NY"



    # Fetch and decode the route
    route_points = get_route(origin, destination)

    # Find intersections along the route
    intersections = await find_intersections(route_points)

    # Cluster intersections using DBSCAN to remove duplicates
    clustered_intersections = cluster_intersections(intersections)

    # Visualize the route with intersections
    visualize_route_with_intersections(route_points, clustered_intersections, origin, destination)


if __name__ == "__main__":
    asyncio.run(main())
