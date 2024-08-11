import requests
import logging
from geopy.distance import geodesic
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from server import models
# Configure logging
logger = logging.getLogger(__name__)

class Vehicle:
    """
    Represents a vehicle with attributes necessary for prioritization.
    """
    def __init__(self, id, type, gps_data, urgency_level, current_speed):
        self.id = id
        self.type = type
        self.gps_data = gps_data
        self.urgency_level = urgency_level
        self.current_speed = current_speed
        self.priority_score = self.assign_priority_score()
        logger.info(f"Vehicle {self.id} created with priority score {self.priority_score}.")

    def assign_priority_score(self):
        """
        Assigns a priority score to the vehicle based on its type and urgency level.
        """
        base_score = 0
        if self.type == 'emergency':
            base_score = 10
        elif self.type == 'public_transport':
            base_score = 5
        else:
            base_score = 1

        return base_score + self.urgency_level

class Intersection:
    """
    Represents an intersection with attributes necessary for signal control.
    """
    def __init__(self, id, location, current_traffic_load, signal_timing):
        self.id = id
        self.location = location
        self.current_traffic_load = current_traffic_load
        self.signal_timing = signal_timing

def calculate_eta(vehicle, intersection):
    """
    Calculates the estimated time of arrival (ETA) for a vehicle to an intersection.
    """
    response = requests.get(f'https://maps.googleapis.com/maps/api/directions/json?origin={vehicle.gps_data["lat"]},{vehicle.gps_data["lon"]}&destination={intersection.location["lat"]},{intersection.location["lon"]}&key=YOUR_API_KEY')
    eta = response.json().get('routes', [{}])[0].get('legs', [{}])[0].get('duration', {}).get('value', 0)
    logger.info(f"ETA for vehicle {vehicle.id} to intersection {intersection.id} is {eta} seconds.")
    return eta

def fetch_real_time_traffic_data():
    """
    Fetches real-time traffic data from an external API.
    """
    response = requests.get('https://api.trafficdata.com/realtime')
    logger.info("Fetched real-time traffic data.")
    return response.json()

def fetch_historical_traffic_data():
    """
    Fetches historical traffic data from an external API.
    """
    response = requests.get('https://api.trafficdata.com/historical')
    logger.info("Fetched historical traffic data.")
    return response.json()

def fetch_weather_data():
    """
    Fetches current weather data from an external API.
    """
    response = requests.get('https://api.weatherdata.com/current')
    logger.info("Fetched weather data.")
    return response.json()

def calculate_weight(vehicle, intersection, real_time_traffic_data, historical_data, weather_data):
    """
    Calculates the weight for signal adjustment based on multiple factors.
    """
    eta = calculate_eta(vehicle, intersection)
    current_traffic_load = real_time_traffic_data.get(intersection.id, 1)
    historical_traffic_pattern = historical_data.get(intersection.id, 1)
    weather_impact = weather_data.get('impact', 1)

    weight = (vehicle.priority_score / eta) * current_traffic_load * historical_traffic_pattern * weather_impact
    logger.info(f"Calculated weight for vehicle {vehicle.id} at intersection {intersection.id} is {weight}.")
    return weight

def adjust_signal_timing(signal_data, weight):
    """
    Adjusts the signal timing based on the calculated weight.
    """
    new_timing = signal_data['default_timing'] * weight
    logger.info(f"Adjusted signal timing to {new_timing}.")
    return new_timing

async def update_signal(signal_id, new_timing, db: AsyncSession):
    """
    Updates the signal timing in the database.
    """
    result = await db.execute(select(models.TrafficSignal).filter(models.TrafficSignal.signal_id == signal_id))
    db_signal = result.scalar_one_or_none()
    if db_signal:
        db_signal.timing = new_timing
        await db.commit()
        await db.refresh(db_signal)
        logger.info(f"Updated signal {signal_id} with new timing {new_timing}.")

async def priority_algorithm(db: AsyncSession):
    """
    Runs the priority algorithm to adjust signal timings based on vehicle priorities and real-time data.
    """
    vehicles = await get_vehicles_from_data()
    intersections = await get_all_intersections(db)
    real_time_traffic_data = fetch_real_time_traffic_data()
    historical_data = fetch_historical_traffic_data()
    weather_data = fetch_weather_data()

    for vehicle in vehicles:
        route = get_vehicle_route(vehicle)
        intersections_in_route = get_intersections_in_route(route)
        for intersection in intersections_in_route:
            weight = calculate_weight(vehicle, intersection, real_time_traffic_data, historical_data, weather_data)
            new_timing = adjust_signal_timing(intersection.signal_timing, weight)
            await update_signal(intersection.id, new_timing, db)

async def get_vehicles_from_data():
    """
    Fetches vehicle data from an external API.
    """
    response = requests.get('https://api.vehiclesdata.com/current')
    vehicles_data = response.json()
    vehicles = [Vehicle(**data) for data in vehicles_data]
    logger.info("Fetched vehicle data.")
    return vehicles

async def get_all_intersections(db: AsyncSession):
    """
    Fetches all intersections from the database.
    """
    result = await db.execute(select(models.TrafficSignal))
    intersections_data = result.scalars().all()
    intersections = [Intersection(id=data.signal_id, location={'lat': data.latitude, 'lon': data.longitude}, current_traffic_load=1, signal_timing=data) for data in intersections_data]
    logger.info("Fetched intersections data.")
    return intersections
