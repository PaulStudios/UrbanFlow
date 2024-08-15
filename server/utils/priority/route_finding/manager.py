import logging
import uuid
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Dict, List
from uuid import UUID

from geopy.distance import geodesic
from rtree import index
from sqlalchemy.ext.asyncio import AsyncSession

from server.cache import cache_manager
from server.utils.priority.basic import calculate_distance
from server.utils.priority.route_finding.intersections import get_route, find_intersections, cluster_intersections, \
    visualize_route_with_intersections, filter_intersections
from server.utils.priority.route_finding.spatial_index import signal_index, initialize_spatial_index, \
    verify_spatial_index

logger = logging.getLogger(__name__)


async def get_vehicle_route(
        origin: str, destination: str,
        vehicle_id: uuid.UUID, db: AsyncSession
) -> Tuple[UUID, Dict[str, List[List | Tuple]]] | bool:
    """
    Calculate the route and intersections for a vehicle, store the data in cache, and return the result.

    Args:
        origin (str): The starting point for the route.
        destination (str): The end point for the route.
        vehicle_id (uuid.UUID): A unique identifier for the vehicle.
        db (AsyncSession): A database session object.

    Returns:
        Tuple[UUID, Dict[str, List[List | Tuple]]] | bool: A tuple containing the `vehicle_id`
        and a dictionary with the route and intersections data. The dictionary has three keys:
            - "route": A list of points representing the route.
            - "intersections": A list of points representing the filtered intersections along the route.
            - "filtered_intersections": A list of intersections that are near traffic signals.
        Returns False if the vehicle ID is invalid.
    """
    logger.info(f"Starting route calculation for vehicle {vehicle_id}")

    if not isinstance(vehicle_id, UUID):
        logger.error(f"Invalid Vehicle ID: {vehicle_id}")
        return False

    logger.info(f"Origin: {origin}, Destination: {destination}")
    cache_manager.set_result(vehicle_id, "PROGRESS")

    # Calculate route and intersections
    route_points = get_route(origin, destination)
    intersections_all = await find_intersections(route_points)
    intersections_clustered = cluster_intersections(intersections_all)
    intersections = filter_intersections(route_points, intersections_clustered)

    visualize_route_with_intersections(route_points, intersections, origin, destination)

    # Retrieve all signals from cache if not already loaded
    if not signal_index.signal_dict:
        logger.info("Retrieving all signals from cache")
        all_signals = await cache_manager.get_all_signals_from_cache(db)
        logger.info(f"Number of signals retrieved: {len(all_signals)}")
        initialize_spatial_index(all_signals)

    verify_spatial_index()
    # Filter intersections near signals
    filtered_intersections = await filter_intersections_near_signals(intersections)

    # Save route data in cache
    logger.info(f"Saving route data for vehicle {vehicle_id}")
    save_status = cache_manager.save_route_data(vehicle_id, route_points, filtered_intersections, intersections)

    # Set cache status based on save result
    if save_status:
        logger.info(f"Route data saved successfully for vehicle {vehicle_id}")
        cache_manager.set_result(vehicle_id, "DONE")
    else:
        logger.error(f"Failed to save route data for vehicle {vehicle_id}")
        cache_manager.set_result(vehicle_id, "ERROR")

    return vehicle_id, {
        "route": list(route_points),
        "intersections": list(intersections),
        "filtered_intersections": list(filtered_intersections),
    }


async def filter_intersections_near_signals(
        intersections: List[Tuple[float, float]],
        radius: float = 50.0
) -> List[Tuple[float, float]]:
    """
    Filter intersections that are near any traffic signals.

    Args:
        intersections (List[Tuple[float, float]]): List of intersection coordinates.
        radius (float): The distance threshold to consider an intersection "near" a signal. Default is 50 meters.

    Returns:
        List[Tuple[float, float]]: A list of intersections that are near signals.
    """
    logger.info(f"Filtering intersections within {radius} meters of signals")
    filtered_intersections = []

    # Increase the buffer size to be more generous
    buffer = (radius / 111000) * 2  # Rough conversion of meters to degrees, doubled for safety

    for intersection in intersections:
        lat, lon = intersection  # Note: lat comes first
        bbox = (lat - buffer, lon - buffer, lat + buffer, lon + buffer)
        nearby_signal_ids = signal_index.query(bbox)

        logger.debug(f"Intersection {intersection} - Nearby signals: {len(nearby_signal_ids)}")

        if nearby_signal_ids:
            nearby_signals = [signal_index.signal_dict[sid] for sid in nearby_signal_ids]

            distances = [geodesic(intersection, signal).meters for signal in nearby_signals]

            min_distance = min(distances) if distances else None
            logger.debug(f"Intersection {intersection} - Minimum distance: {min_distance}")

            if any(d <= radius for d in distances):
                filtered_intersections.append(intersection)
                logger.info(f"Intersection {intersection} is near a signal (distance: {min_distance}m)")
        else:
            logger.debug(f"No nearby signals found for intersection {intersection}")

    logger.info(f"Total filtered intersections: {len(filtered_intersections)}")

    # Log all signals in the index for verification
    logger.debug(f"All signals in index: {list(signal_index.signal_dict.items())}")

    return filtered_intersections
