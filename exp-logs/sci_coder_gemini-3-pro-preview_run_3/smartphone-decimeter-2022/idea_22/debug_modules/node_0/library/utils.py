import os
import numpy as np
import pandas as pd
from library.config import (
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
)

# --- WGS84 Ellipsoid Constants ---
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # Square of eccentricity


def wgs84_to_ecef(lat, lon, alt):
    """
    Convert WGS84 Geodetic coordinates to ECEF.

    Args:
        lat: Latitude in degrees (float or np.array)
        lon: Longitude in degrees (float or np.array)
        alt: Altitude in meters (float or np.array)

    Returns:
        x, y, z: ECEF coordinates in meters
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)

    x = (n + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (n + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (n * (1 - WGS84_E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_wgs84(x, y, z):
    """
    Convert ECEF coordinates to WGS84 Geodetic coordinates.
    Uses Bowring's method for high precision.

    Args:
        x, y, z: ECEF coordinates in meters (float or np.array)

    Returns:
        lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m)
    """
    # Ensure inputs are numpy arrays for vectorization
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Second eccentricity squared
    e2_prime = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2

    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)

    lon_rad = np.arctan2(y, x)

    lat_num = z + e2_prime * WGS84_B * np.sin(theta) ** 3
    lat_denom = p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3
    lat_rad = np.arctan2(lat_num, lat_denom)

    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)
    alt = p / np.cos(lat_rad) - n

    return np.rad2deg(lat_rad), np.rad2deg(lon_rad), alt


def ecef_to_enu(x, y, z, lat0, lon0, alt0):
    """
    Convert ECEF coordinates to local East-North-Up (ENU) frame.

    Args:
        x, y, z: Target ECEF coordinates (meters)
        lat0, lon0, alt0: Reference origin in WGS84 (deg, deg, m)

    Returns:
        e, n, u: ENU coordinates in meters
    """
    # Convert origin to ECEF
    x0, y0, z0 = wgs84_to_ecef(lat0, lon0, alt0)

    # Calculate difference vector
    dx = x - x0
    dy = y - y0
    dz = z - z0

    # Rotation matrix parameters
    phi = np.deg2rad(lat0)
    lam = np.deg2rad(lon0)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Rotation matrix R (ECEF to ENU)
    # [ -sin(lam)           cos(lam)          0       ]
    # [ -sin(phi)cos(lam)  -sin(phi)sin(lam)  cos(phi)]
    # [  cos(phi)cos(lam)   cos(phi)sin(lam)  sin(phi)]

    e = -sin_lam * dx + cos_lam * dy
    n = -sin_phi * cos_lam * dx - sin_phi * sin_lam * dy + cos_phi * dz
    u = cos_phi * cos_lam * dx + cos_phi * sin_lam * dy + sin_phi * dz

    return e, n, u


def enu_to_ecef(e, n, u, lat0, lon0, alt0):
    """
    Convert local ENU coordinates to ECEF frame.

    Args:
        e, n, u: Local ENU coordinates (meters)
        lat0, lon0, alt0: Reference origin in WGS84 (deg, deg, m)

    Returns:
        x, y, z: ECEF coordinates in meters
    """
    # Convert origin to ECEF
    x0, y0, z0 = wgs84_to_ecef(lat0, lon0, alt0)

    # Rotation matrix parameters
    phi = np.deg2rad(lat0)
    lam = np.deg2rad(lon0)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Inverse rotation (Transpose of R)
    # [ -sin(lam)  -sin(phi)cos(lam)   cos(phi)cos(lam) ]
    # [  cos(lam)  -sin(phi)sin(lam)   cos(phi)sin(lam) ]
    # [     0           cos(phi)           sin(phi)     ]

    dx = -sin_lam * e - sin_phi * cos_lam * n + cos_phi * cos_lam * u
    dy = cos_lam * e - sin_phi * sin_lam * n + cos_phi * sin_lam * u
    dz = cos_phi * n + sin_phi * u

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    return x, y, z


def process_with_cache(filename, processing_func, load_cached_data=True, **func_kwargs):
    """
    Executes a processing function with strict caching logic using Parquet.

    Args:
        filename (str): Name of the cache file (e.g., 'data.parquet').
        processing_func (callable): Function to compute data if cache is missed.
                                    Must return a pandas DataFrame.
        load_cached_data (bool): Whether to attempt loading from cache.
        **func_kwargs: Arguments to pass to processing_func.

    Returns:
        pd.DataFrame: The processed data.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    file_path = os.path.join(WORKING_DIR, filename)

    # 1. Try to load
    if load_cached_data:
        if os.path.exists(file_path):
            try:
                print(f"Loading cached data from {file_path}...")
                df = pd.read_parquet(file_path)
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        else:
            print(f"Cache file {file_path} not found. Recomputing...")
    else:
        print("Force recompute requested...")

    # 2. Compute
    print("Executing processing function...")
    df = processing_func(**func_kwargs)

    # 3. Save
    try:
        print(f"Saving data to cache: {file_path}")
        df.to_parquet(file_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {file_path}: {e}")

    return df


def load_metadata(split="train"):
    """
    Load metadata for a specific split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    if split == "train":
        path = TRAIN_METADATA_PATH
    elif split == "val":
        path = VAL_METADATA_PATH
    elif split == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Metadata file not found at {path}. Please run metadata generation first."
        )

    return pd.read_csv(path)
