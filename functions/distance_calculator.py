"""Calculate the great circle distance in kilometers between two points"""
import numpy as np

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance in kilometers between two points
    on the Earth specified by their latitude and longitude in decimal degrees.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    half_chord_square = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    angular_distance = 2 * np.arcsin(np.sqrt(half_chord_square))

    # Radius of Earth in kilometers (KM)
    radius = 6371.0
    return angular_distance * radius
