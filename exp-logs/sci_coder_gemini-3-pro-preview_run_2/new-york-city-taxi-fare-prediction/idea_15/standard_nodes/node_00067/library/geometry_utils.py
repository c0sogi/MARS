import numpy as np
from library.config import Config


class DistanceCalculator:
    """
    Provides vectorized geometric calculations for spatial features.
    Includes Haversine (Great Circle), Manhattan (L1), and Coordinate Rotation.
    """

    EARTH_RADIUS_KM = 6371.0

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points in kilometers.
        Vectorized for NumPy arrays.

        Args:
            lat1, lon1: Origin coordinates (decimal degrees).
            lat2, lon2: Destination coordinates (decimal degrees).

        Returns:
            np.ndarray: Distance in kilometers.
        """
        # Ensure inputs are numpy arrays
        lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])

        # Convert decimal degrees to radians
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
            np.radians, [lat1, lon1, lat2, lon2]
        )

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        return DistanceCalculator.EARTH_RADIUS_KM * c

    @staticmethod
    def manhattan(lat1, lon1, lat2, lon2):
        """
        Calculate the Manhattan distance (L1 norm) approximation in kilometers.
        Useful for city-grid based travel estimation.

        Args:
            lat1, lon1: Origin coordinates (decimal degrees).
            lat2, lon2: Destination coordinates (decimal degrees).

        Returns:
            np.ndarray: Manhattan distance in kilometers.
        """
        lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])

        # Approximation factors:
        # 1 deg lat ~= 111.32 km
        # 1 deg lon ~= 111.32 * cos(lat) km

        avg_lat_rad = np.radians((lat1 + lat2) / 2.0)

        # Calculate displacements in km
        lat_diff_km = np.abs(lat1 - lat2) * 111.32
        lon_diff_km = np.abs(lon1 - lon2) * 111.32 * np.cos(avg_lat_rad)

        return lat_diff_km + lon_diff_km

    @staticmethod
    def rotated_coordinates(lat, lon, angle_deg=29.0):
        """
        Rotate coordinates to align with the NYC street grid.
        NYC grid is approximately rotated 29 degrees from True North.

        Args:
            lat, lon: Coordinates (decimal degrees).
            angle_deg: Rotation angle in degrees (default 29.0 for NYC).

        Returns:
            tuple: (rotated_lat, rotated_lon)
        """
        lat, lon = map(np.asarray, [lat, lon])

        angle_rad = np.radians(angle_deg)
        sin_a = np.sin(angle_rad)
        cos_a = np.cos(angle_rad)

        # Rotation matrix application:
        # x' = x * cos(theta) - y * sin(theta)
        # y' = x * sin(theta) + y * cos(theta)
        # Mapping: x -> lon, y -> lat

        lon_rot = lon * cos_a - lat * sin_a
        lat_rot = lon * sin_a + lat * cos_a

        return lat_rot, lon_rot


class GridIndexer:
    """
    Handles spatial discretization (binning) of coordinates into string keys.
    Simulates Geohash functionality using configurable grid sizes.
    """

    @staticmethod
    def get_grid_key(lat, lon, precision_key):
        """
        Generates a unique string key for a coordinate pair based on grid precision.
        Used for creating 'Spatial Base' priors.

        Args:
            lat (np.ndarray): Latitude values.
            lon (np.ndarray): Longitude values.
            precision_key (str): Key mapping to Config.GRID_SIZES (e.g., 'L5', 'L6').

        Returns:
            np.ndarray: Array of string keys (e.g., "900_-1600").
        """
        if precision_key not in Config.GRID_SIZES:
            raise ValueError(
                f"Precision key '{precision_key}' not found in Config.GRID_SIZES"
            )

        step = Config.GRID_SIZES[precision_key]

        # Ensure inputs are numpy arrays
        lat = np.asarray(lat)
        lon = np.asarray(lon)

        # Discretize coordinates (Binning)
        # Using floor to determine the grid cell index
        lat_idx = np.floor(lat / step).astype(int)
        lon_idx = np.floor(lon / step).astype(int)

        # Vectorized string construction
        # Format: "{lat_idx}_{lon_idx}"
        lat_str = lat_idx.astype(str)
        lon_str = lon_idx.astype(str)

        # Create separator array for vectorized concatenation
        # Handling scalar vs array input gracefully
        if lat_str.ndim == 0:
            return f"{lat_str}_{lon_str}"

        sep = np.array(["_"] * len(lat_str), dtype="U1")

        # Concatenate: lat + "_" + lon
        keys = np.char.add(np.char.add(lat_str, sep), lon_str)

        return keys

    @staticmethod
    def clamp_coordinates(lat, lon):
        """
        Clamps coordinates to the NYC bounding box defined in Config.
        Prevents outliers from generating invalid grid keys or skewing stats.

        Args:
            lat, lon: Input coordinates.

        Returns:
            tuple: (clamped_lat, clamped_lon)
        """
        lat = np.asarray(lat)
        lon = np.asarray(lon)

        lat_clamped = np.clip(lat, Config.NYC_LAT_MIN, Config.NYC_LAT_MAX)
        lon_clamped = np.clip(lon, Config.NYC_LON_MIN, Config.NYC_LON_MAX)

        return lat_clamped, lon_clamped
