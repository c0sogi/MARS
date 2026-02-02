import numpy as np
from library.config import Config


class GeoUtils:
    """
    Utility class for geospatial calculations and coordinate transformations.
    Uses WGS84 constants defined in the Config class.
    """

    @staticmethod
    def lla_to_ecef(lat, lon, alt):
        """
        Convert Latitude, Longitude, Altitude to ECEF coordinates.

        Args:
            lat (np.array or float): Latitude in degrees.
            lon (np.array or float): Longitude in degrees.
            alt (np.array or float): Altitude in meters.

        Returns:
            tuple: (x, y, z) ECEF coordinates in meters.
        """
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        a = Config.WGS84_A
        e2 = Config.WGS84_E2

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        # Radius of curvature in the prime vertical
        N = a / np.sqrt(1 - e2 * sin_lat**2)

        x = (N + alt) * cos_lat * cos_lon
        y = (N + alt) * cos_lat * sin_lon
        z = (N * (1 - e2) + alt) * sin_lat

        return x, y, z

    @staticmethod
    def ecef_to_lla(x, y, z):
        """
        Convert ECEF coordinates to Latitude, Longitude, Altitude.
        Uses Heikkinen's exact solution (closed-form).

        Args:
            x (np.array or float): X coordinate in meters.
            y (np.array or float): Y coordinate in meters.
            z (np.array or float): Z coordinate in meters.

        Returns:
            tuple: (lat, lon, alt) in degrees and meters.
        """
        a = Config.WGS84_A
        b = Config.WGS84_B
        e2 = Config.WGS84_E2
        ep2 = (a**2 - b**2) / b**2

        p = np.sqrt(x**2 + y**2)
        theta = np.arctan2(z * a, p * b)

        lon_rad = np.arctan2(y, x)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        lat_rad = np.arctan2(z + ep2 * b * sin_theta**3, p - e2 * a * cos_theta**3)

        sin_lat = np.sin(lat_rad)
        N = a / np.sqrt(1 - e2 * sin_lat**2)

        # Calculate altitude.
        # For latitudes close to the poles, using z/sin(lat) is numerically more stable,
        # but for typical smartphone data (mid-latitudes), p/cos(lat) is fine.
        # We use a hybrid approach or just the standard one which is generally sufficient.
        alt = p / np.cos(lat_rad) - N

        lat = np.rad2deg(lat_rad)
        lon = np.rad2deg(lon_rad)

        return lat, lon, alt

    @staticmethod
    def ecef_to_enu(x, y, z, lat0, lon0, alt0):
        """
        Convert ECEF coordinates to ENU (East, North, Up) relative to a reference point.

        Args:
            x, y, z (np.array or float): ECEF coordinates of points to convert.
            lat0, lon0, alt0 (float): Reference LLA coordinates (anchor).

        Returns:
            tuple: (e, n, u) ENU coordinates in meters.
        """
        # Convert reference point to ECEF
        x0, y0, z0 = GeoUtils.lla_to_ecef(lat0, lon0, alt0)

        # Delta ECEF
        dx = x - x0
        dy = y - y0
        dz = z - z0

        lat0_rad = np.deg2rad(lat0)
        lon0_rad = np.deg2rad(lon0)

        sin_lat = np.sin(lat0_rad)
        cos_lat = np.cos(lat0_rad)
        sin_lon = np.sin(lon0_rad)
        cos_lon = np.cos(lon0_rad)

        # Rotation matrix application
        e = -sin_lon * dx + cos_lon * dy
        n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

        return e, n, u

    @staticmethod
    def enu_to_ecef(e, n, u, lat0, lon0, alt0):
        """
        Convert ENU coordinates to ECEF relative to a reference point.

        Args:
            e, n, u (np.array or float): ENU coordinates.
            lat0, lon0, alt0 (float): Reference LLA coordinates (anchor).

        Returns:
            tuple: (x, y, z) ECEF coordinates.
        """
        x0, y0, z0 = GeoUtils.lla_to_ecef(lat0, lon0, alt0)

        lat0_rad = np.deg2rad(lat0)
        lon0_rad = np.deg2rad(lon0)

        sin_lat = np.sin(lat0_rad)
        cos_lat = np.cos(lat0_rad)
        sin_lon = np.sin(lon0_rad)
        cos_lon = np.cos(lon0_rad)

        # Inverse rotation (transpose of the rotation matrix)
        dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
        dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
        dz = cos_lat * n + sin_lat * u

        x = x0 + dx
        y = y0 + dy
        z = z0 + dz

        return x, y, z

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points on the earth (specified in decimal degrees).

        Args:
            lat1, lon1: First point coordinates.
            lat2, lon2: Second point coordinates.

        Returns:
            Distance in meters.
        """
        # Convert decimal degrees to radians
        lat1_rad = np.deg2rad(lat1)
        lon1_rad = np.deg2rad(lon1)
        lat2_rad = np.deg2rad(lat2)
        lon2_rad = np.deg2rad(lon2)

        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        # Mean Earth Radius in meters
        r = 6371000
        return c * r

    @staticmethod
    def calculate_bearing(lat1, lon1, lat2, lon2):
        """
        Calculate the bearing between two points.

        Args:
            lat1, lon1: Start point coordinates.
            lat2, lon2: End point coordinates.

        Returns:
            Bearing in degrees (0-360).
        """
        lat1_rad = np.deg2rad(lat1)
        lon1_rad = np.deg2rad(lon1)
        lat2_rad = np.deg2rad(lat2)
        lon2_rad = np.deg2rad(lon2)

        dlon = lon2_rad - lon1_rad

        x = np.sin(dlon) * np.cos(lat2_rad)
        y = np.cos(lat1_rad) * np.sin(lat2_rad) - (
            np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        )

        initial_bearing = np.arctan2(x, y)

        # Convert to degrees and normalize to 0-360
        initial_bearing = np.rad2deg(initial_bearing)
        bearing = (initial_bearing + 360) % 360

        return bearing

    @staticmethod
    def get_rotation_matrix_ecef_to_enu(lat, lon):
        """
        Compute the rotation matrix from ECEF to ENU frame at a given lat/lon.
        Useful for rotating velocity vectors.

        Args:
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.

        Returns:
            np.array: 3x3 Rotation matrix.
        """
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        # Row 1: East unit vector in ECEF
        # Row 2: North unit vector in ECEF
        # Row 3: Up unit vector in ECEF
        R = np.array(
            [
                [-sin_lon, cos_lon, 0],
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
            ]
        )
        return R
