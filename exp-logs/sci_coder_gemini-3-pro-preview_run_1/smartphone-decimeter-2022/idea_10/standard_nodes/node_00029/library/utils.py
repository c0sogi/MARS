import os
import numpy as np
import pandas as pd
from library.config import Config

# =============================================================================
# WGS84 Coordinate Transformations
# =============================================================================


def llh_to_ecef(lat, lon, alt):
    """
    Convert Latitude, Longitude, Altitude to ECEF (X, Y, Z).
    lat, lon in degrees. alt in meters.
    Vectorized for numpy arrays.
    """
    a = Config.WGS84_A
    b = Config.WGS84_B
    e2 = 1 - (b**2 / a**2)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_llh(x, y, z):
    """
    Convert ECEF (X, Y, Z) to Latitude, Longitude, Altitude.
    Returns lat, lon in degrees, alt in meters.
    Using Ferrari's method (closed-form approximation).
    Vectorized for numpy arrays.
    """
    a = Config.WGS84_A
    b = Config.WGS84_B
    e2 = 1 - (b**2 / a**2)
    ep2 = (a**2 - b**2) / b**2

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)

    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    return np.degrees(lat), np.degrees(lon), alt


def ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt):
    """
    Convert ECEF coordinates to ENU relative to a reference point (ref_lat, ref_lon, ref_alt).
    Vectorized for numpy arrays.
    """
    # Reference point in ECEF
    ref_x, ref_y, ref_z = llh_to_ecef(ref_lat, ref_lon, ref_alt)

    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    # Rotation matrix components
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # E = -sin(lon)*dx + cos(lon)*dy
    e = -sin_lon * dx + cos_lon * dy

    # N = -sin(lat)cos(lon)*dx - sin(lat)sin(lon)*dy + cos(lat)*dz
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz

    # U = cos(lat)cos(lon)*dx + cos(lat)sin(lon)*dy + sin(lat)*dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt):
    """
    Convert ENU coordinates to ECEF relative to a reference point.
    Vectorized for numpy arrays.
    """
    ref_x, ref_y, ref_z = llh_to_ecef(ref_lat, ref_lon, ref_alt)

    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Inverse rotation
    # dx = -sin(lon)*E - sin(lat)cos(lon)*N + cos(lat)cos(lon)*U
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u

    # dy = cos(lon)*E - sin(lat)sin(lon)*N + cos(lat)sin(lon)*U
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u

    # dz = cos(lat)*N + sin(lat)*U
    dz = cos_lat * n + sin_lat * u

    return ref_x + dx, ref_y + dy, ref_z + dz


# =============================================================================
# Metric Calculation
# =============================================================================


def haversine_np(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) using numpy.
    Returns distance in meters.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of earth in meters (mean radius)
    r = 6371000
    return c * r


def calc_score(df_pred, df_gt):
    """
    Calculate the competition metric: mean of the 50th and 95th percentile distance errors.

    Args:
        df_pred: DataFrame with columns ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']
        df_gt: DataFrame with columns ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']
               Note: tripId is constructed from drive_id and phone_name if not present.
    """
    # Ensure tripId exists in GT if not present (construct from drive_id and phone_name)
    if (
        "tripId" not in df_gt.columns
        and "drive_id" in df_gt.columns
        and "phone_name" in df_gt.columns
    ):
        df_gt = df_gt.copy()
        df_gt["tripId"] = df_gt["drive_id"] + "-" + df_gt["phone_name"]

    # Merge predictions and ground truth
    df = pd.merge(
        df_pred, df_gt, on=["tripId", "UnixTimeMillis"], suffixes=("_pred", "_gt")
    )

    # Calculate distance error
    df["dist"] = haversine_np(
        df["LatitudeDegrees_pred"],
        df["LongitudeDegrees_pred"],
        df["LatitudeDegrees_gt"],
        df["LongitudeDegrees_gt"],
    )

    # Extract phone name from tripId (format: drive_id-phone_name)
    # Phone name is the part after the last hyphen
    df["phone"] = df["tripId"].apply(lambda x: x.split("-")[-1])

    # Calculate percentiles per phone
    scores = []
    for phone, group in df.groupby("phone"):
        p50 = np.percentile(group["dist"], 50)
        p95 = np.percentile(group["dist"], 95)
        scores.append((p50 + p95) / 2)

    # Mean across all phones
    return np.mean(scores)


# =============================================================================
# Data Processing with Caching
# =============================================================================


def process_target_residuals(metadata_df, load_cached_data=True):
    """
    Computes ENU residuals (target variables) for the training data.

    Logic:
    1. Load WLS baseline (from device_gnss.csv).
    2. Load Ground Truth (from metadata).
    3. Convert WLS ECEF -> WLS LLH.
    4. Convert GT LLH -> GT ECEF (using WLS Altitude to isolate horizontal error).
    5. Calculate Delta ECEF (GT - WLS).
    6. Rotate Delta ECEF to ENU using WLS LLH as reference.
    7. Return DataFrame with ['drive_id', 'phone_name', 'UnixTimeMillis', 'dLat_meters', 'dLon_meters'].

    Caching:
    Saves/Loads from Config.WORKING_DIR/target_residuals.parquet
    """
    cache_path = os.path.join(Config.WORKING_DIR, "target_residuals.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached target residuals from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Computing target residuals from scratch...")

    results = []

    # Unique drive-phone pairs
    pairs = metadata_df[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()

    for _, row in pairs.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_rel_path = row["gnss_path"]

        # Load GNSS (Baseline)
        gnss_path = os.path.join(Config.INPUT_DIR, gnss_rel_path)
        if not os.path.exists(gnss_path):
            continue

        try:
            # We only need WLS positions and Time
            cols = [
                "utcTimeMillis",
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
            df_gnss = pd.read_csv(gnss_path, usecols=cols)

            # Filter valid WLS
            df_gnss = df_gnss.dropna(subset=["WlsPositionXEcefMeters"])

            # Get GT for this trip
            df_gt = metadata_df[
                (metadata_df["drive_id"] == drive_id)
                & (metadata_df["phone_name"] == phone_name)
            ].copy()

            if df_gt.empty:
                continue

            # Merge on Time (utcTimeMillis in GNSS ~ UnixTimeMillis in GT)
            merged = pd.merge(
                df_gt,
                df_gnss,
                left_on="UnixTimeMillis",
                right_on="utcTimeMillis",
                how="inner",
            )

            if merged.empty:
                continue

            # 1. Get WLS LLH (Reference for ENU)
            wls_x = merged["WlsPositionXEcefMeters"].values
            wls_y = merged["WlsPositionYEcefMeters"].values
            wls_z = merged["WlsPositionZEcefMeters"].values

            wls_lat, wls_lon, wls_alt = ecef_to_llh(wls_x, wls_y, wls_z)

            # 2. Get GT ECEF
            gt_lat = merged["LatitudeDegrees"].values
            gt_lon = merged["LongitudeDegrees"].values
            # Use WLS altitude for GT ECEF conversion to isolate horizontal error
            # This assumes the vertical error is handled separately or irrelevant for 2D metric
            gt_alt = wls_alt

            gt_x, gt_y, gt_z = llh_to_ecef(gt_lat, gt_lon, gt_alt)

            # 3. Calculate ECEF Delta
            dx = gt_x - wls_x
            dy = gt_y - wls_y
            dz = gt_z - wls_z

            # 4. Rotate to ENU
            # We need to do this point-by-point because reference (WLS) changes
            lat_rad = np.radians(wls_lat)
            lon_rad = np.radians(wls_lon)

            sin_lat = np.sin(lat_rad)
            cos_lat = np.cos(lat_rad)
            sin_lon = np.sin(lon_rad)
            cos_lon = np.cos(lon_rad)

            d_east = -sin_lon * dx + cos_lon * dy
            d_north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz

            # Store
            res_df = pd.DataFrame(
                {
                    "drive_id": drive_id,
                    "phone_name": phone_name,
                    "UnixTimeMillis": merged["UnixTimeMillis"],
                    "dLat_meters": d_north,  # North offset
                    "dLon_meters": d_east,  # East offset
                }
            )

            results.append(res_df)

        except Exception as e:
            print(f"Error processing {drive_id}-{phone_name}: {e}")
            continue

    if not results:
        return pd.DataFrame(
            columns=[
                "drive_id",
                "phone_name",
                "UnixTimeMillis",
                "dLat_meters",
                "dLon_meters",
            ]
        )

    final_df = pd.concat(results, ignore_index=True)

    # Save to cache
    final_df.to_parquet(cache_path, index=False)
    print(f"Saved target residuals to {cache_path}")

    return final_df
