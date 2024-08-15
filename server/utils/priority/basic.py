import logging
import random
import uuid

from geopy import Nominatim
from geopy.distance import geodesic


def get_coordinates(place_name):
    geolocator = Nominatim(user_agent="UrbanFlow/1.0")
    location = geolocator.geocode(place_name)
    if location:
        return location.latitude, location.longitude
    else:
        return None


def calculate_distance_km(coord1, coord2):
    return geodesic(coord1, coord2).kilometers


def calculate_distance(args):
    intersection, signal_coords = args
    return geodesic(intersection, signal_coords).meters


def calculate_approx_time(distance_km):
    speed_km_per_min = 28 / 3
    base_time_minutes = distance_km / speed_km_per_min
    random_factor = random.uniform(0.9, 1.2)  # Random factor between 90% to 120% of the base time
    approx_time_minutes = base_time_minutes * random_factor
    return approx_time_minutes


def uuid_to_hash_int(uuid_str):
    uuid_obj = uuid.UUID(uuid_str)
    hash_value = hash(uuid_obj)
    return abs(hash_value) % (2 ** 31)
