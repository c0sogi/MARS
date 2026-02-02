import numpy as np
import pandas as pd
import os

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 1 - (WGS84_B**2 / WGS84_A**2)  # First eccentricity squared


def WGS84_to_ECEF(lat, lon, alt):
    """
    Convert WGS84 Geodetic coordinates to ECEF.
    Args:
        lat: Latitude in degrees (float or numpy array)
        lon: Longitude in degrees (float or numpy array)
        alt: Altitude in meters (float or numpy array)
    Returns:
        x, y, z: ECEF coordinates in meters
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ECEF_to_WGS84(x, y, z):
    """
    Convert ECEF coordinates to WGS84 Geodetic coordinates using iterative method.
    Args:
        x, y, z: ECEF coordinates in meters
    Returns:
        lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m)
    """
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)

    # Initial guess
    lat = np.arctan2(z, p * (1 - WGS84_E2))

    # Iterative refinement for latitude and altitude
    for _ in range(5):
        sin_lat = np.sin(lat)
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - WGS84_E2 * N / (N + alt)))

    return np.rad2deg(lat), np.rad2deg(lon), alt


def ECEF_to_ENU(x, y, z, ref_lat, ref_lon, ref_alt):
    """
    Convert ECEF coordinates to Local ENU coordinates relative to a reference point.
    Args:
        x, y, z: Target ECEF coordinates
        ref_lat, ref_lon, ref_alt: Reference point WGS84 coordinates
    Returns:
        e, n, u: East, North, Up coordinates in meters
    """
    # Convert reference point to ECEF
    xr, yr, zr = WGS84_to_ECEF(ref_lat, ref_lon, ref_alt)

    dx = x - xr
    dy = y - yr
    dz = z - zr

    ref_lat_rad = np.deg2rad(ref_lat)
    ref_lon_rad = np.deg2rad(ref_lon)

    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)

    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def ENU_to_ECEF(e, n, u, ref_lat, ref_lon, ref_alt):
    """
    Convert Local ENU coordinates to ECEF coordinates relative to a reference point.
    Args:
        e, n, u: ENU coordinates in meters
        ref_lat, ref_lon, ref_alt: Reference point WGS84 coordinates
    Returns:
        x, y, z: ECEF coordinates
    """
    xr, yr, zr = WGS84_to_ECEF(ref_lat, ref_lon, ref_alt)

    ref_lat_rad = np.deg2rad(ref_lat)
    ref_lon_rad = np.deg2rad(ref_lon)

    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)

    # Inverse rotation matrix (Transpose) applied to [e, n, u]
    # R = [[-sin_lon, cos_lon, 0], [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat], [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]]
    # [dx, dy, dz] = R.T @ [e, n, u]

    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    x = xr + dx
    y = yr + dy
    z = zr + dz

    return x, y, z


def ENU_to_WGS84(e, n, u, ref_lat, ref_lon, ref_alt):
    """
    Convert Local ENU coordinates directly to WGS84.
    """
    x, y, z = ENU_to_ECEF(e, n, u, ref_lat, ref_lon, ref_alt)
    return ECEF_to_WGS84(x, y, z)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees).
    """
    R = 6371000.0  # Radius of earth in meters

    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


class ENUTransformer:
    """
    Stateful transformer to handle ENU conversions for a specific reference point.
    """

    def __init__(self, ref_lat, ref_lon, ref_alt=0.0):
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt

    def transform(self, lat, lon, alt=0.0):
        """WGS84 to ENU"""
        x, y, z = WGS84_to_ECEF(lat, lon, alt)
        return ECEF_to_ENU(x, y, z, self.ref_lat, self.ref_lon, self.ref_alt)

    def inverse_transform(self, e, n, u=0.0):
        """ENU to WGS84"""
        return ENU_to_WGS84(e, n, u, self.ref_lat, self.ref_lon, self.ref_alt)


def precompute_enu_coordinates(
    df, lat_col, lon_col, alt_col, cache_key, load_cached_data=True
):
    """
    Computes ENU coordinates for a dataframe with caching mechanism.

    Args:
        df: Input DataFrame containing WGS84 coordinates.
        lat_col, lon_col, alt_col: Column names for lat, lon, alt.
        cache_key: Unique identifier for the cache file (e.g., drive_id + suffix).
        load_cached_data: Boolean to enable/disable loading from cache.

    Returns:
        DataFrame with added 'x_enu', 'y_enu', 'z_enu' columns.
    """
    cache_dir = "./working/idea_18/"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}_enu.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached ENU data from {cache_file}")
        try:
            cached_df = pd.read_parquet(cache_file)
            # Ensure index alignment if necessary, or just return cached if it contains all info
            # Here we assume cached_df is the full result
            return cached_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute
    print(f"Computing ENU coordinates for {cache_key}...")

    # Use the first point as reference for the whole dataframe if it represents a single sequence
    # Or use mean. Here we use the first valid point.
    ref_lat = df[lat_col].iloc[0]
    ref_lon = df[lon_col].iloc[0]
    ref_alt = df[alt_col].iloc[0] if alt_col in df.columns else 0.0

    # Handle missing altitude if column doesn't exist or has NaNs
    alts = (
        df[alt_col].fillna(0.0).values if alt_col in df.columns else np.zeros(len(df))
    )

    lats = df[lat_col].values
    lons = df[lon_col].values

    x, y, z = WGS84_to_ECEF(lats, lons, alts)
    e, n, u = ECEF_to_ENU(x, y, z, ref_lat, ref_lon, ref_alt)

    result_df = df.copy()
    result_df["x_enu"] = e
    result_df["y_enu"] = n
    result_df["z_enu"] = u
    result_df["ref_lat"] = ref_lat
    result_df["ref_lon"] = ref_lon
    result_df["ref_alt"] = ref_alt

    # Save to cache
    print(f"Saving ENU data to {cache_file}")
    result_df.to_parquet(cache_file)

    return result_df
