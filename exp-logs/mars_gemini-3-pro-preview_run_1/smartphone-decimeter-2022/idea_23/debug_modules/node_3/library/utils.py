import numpy as np


class GeodeticUtils:
    """
    Utility class for geodetic coordinate transformations and distance calculations.
    Uses the WGS84 ellipsoid model.
    """

    # WGS84 Ellipsoid Constants
    A = 6378137.0  # Semi-major axis (meters)
    F = 1 / 298.257223563  # Flattening
    B = A * (1 - F)  # Semi-minor axis (meters)
    E2 = 2 * F - F**2  # Eccentricity squared
    E2_PRIME = E2 / (1 - E2)  # Second eccentricity squared

    @staticmethod
    def geodetic_to_ecef(lat, lon, alt):
        """
        Convert geodetic coordinates (WGS84) to ECEF coordinates.

        Args:
            lat (float or np.array): Latitude in degrees.
            lon (float or np.array): Longitude in degrees.
            alt (float or np.array): Altitude in meters.

        Returns:
            tuple: (x, y, z) in ECEF coordinates (meters).
        """
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        n = GeodeticUtils.A / np.sqrt(1 - GeodeticUtils.E2 * np.sin(lat_rad) ** 2)

        x = (n + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (n + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (n * (1 - GeodeticUtils.E2) + alt) * np.sin(lat_rad)

        return x, y, z

    @staticmethod
    def ecef_to_geodetic(x, y, z):
        """
        Convert ECEF coordinates to geodetic coordinates (WGS84).
        Uses Ferrari's solution.

        Args:
            x (float or np.array): X coordinate in meters.
            y (float or np.array): Y coordinate in meters.
            z (float or np.array): Z coordinate in meters.

        Returns:
            tuple: (lat, lon, alt) in degrees and meters.
        """
        p = np.sqrt(x**2 + y**2)
        theta = np.arctan2(z * GeodeticUtils.A, p * GeodeticUtils.B)

        lon_rad = np.arctan2(y, x)

        lat_num = z + GeodeticUtils.E2_PRIME * GeodeticUtils.B * np.sin(theta) ** 3
        lat_den = p - GeodeticUtils.E2 * GeodeticUtils.A * np.cos(theta) ** 3
        lat_rad = np.arctan2(lat_num, lat_den)

        n = GeodeticUtils.A / np.sqrt(1 - GeodeticUtils.E2 * np.sin(lat_rad) ** 2)
        alt = p / np.cos(lat_rad) - n

        return np.rad2deg(lat_rad), np.rad2deg(lon_rad), alt

    @staticmethod
    def wgs84_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref):
        """
        Convert WGS84 coordinates to local ENU coordinates relative to a reference point.

        Args:
            lat, lon, alt: Target geodetic coordinates (degrees, meters).
            lat_ref, lon_ref, alt_ref: Reference geodetic coordinates (degrees, meters).

        Returns:
            tuple: (e, n, u) in meters.
        """
        # Convert both to ECEF
        x, y, z = GeodeticUtils.geodetic_to_ecef(lat, lon, alt)
        xr, yr, zr = GeodeticUtils.geodetic_to_ecef(lat_ref, lon_ref, alt_ref)

        dx = x - xr
        dy = y - yr
        dz = z - zr

        lat_ref_rad = np.deg2rad(lat_ref)
        lon_ref_rad = np.deg2rad(lon_ref)

        sin_lat = np.sin(lat_ref_rad)
        cos_lat = np.cos(lat_ref_rad)
        sin_lon = np.sin(lon_ref_rad)
        cos_lon = np.cos(lon_ref_rad)

        e = -sin_lon * dx + cos_lon * dy
        n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

        return e, n, u

    @staticmethod
    def enu_to_wgs84(e, n, u, lat_ref, lon_ref, alt_ref):
        """
        Convert local ENU coordinates to WGS84 coordinates relative to a reference point.

        Args:
            e, n, u: Local coordinates in meters.
            lat_ref, lon_ref, alt_ref: Reference geodetic coordinates (degrees, meters).

        Returns:
            tuple: (lat, lon, alt) in degrees and meters.
        """
        lat_ref_rad = np.deg2rad(lat_ref)
        lon_ref_rad = np.deg2rad(lon_ref)

        sin_lat = np.sin(lat_ref_rad)
        cos_lat = np.cos(lat_ref_rad)
        sin_lon = np.sin(lon_ref_rad)
        cos_lon = np.cos(lon_ref_rad)

        # Rotation matrix inverse (transpose)
        # | -sin_lon  -sin_lat*cos_lon   cos_lat*cos_lon |
        # |  cos_lon  -sin_lat*sin_lon   cos_lat*sin_lon |
        # |     0          cos_lat           sin_lat     |

        dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
        dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
        dz = cos_lat * n + sin_lat * u

        xr, yr, zr = GeodeticUtils.geodetic_to_ecef(lat_ref, lon_ref, alt_ref)

        x = xr + dx
        y = yr + dy
        z = zr + dz

        return GeodeticUtils.ecef_to_geodetic(x, y, z)

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points on the earth (specified in decimal degrees).

        Args:
            lat1, lon1: First point coordinates in degrees.
            lat2, lon2: Second point coordinates in degrees.

        Returns:
            float or np.array: Distance in meters.
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371000  # Radius of earth in meters
        return c * r
