import numpy as np


def wgs84_to_local_meters(lat_ref, lon_ref, lat, lon):
    """
    Converts target (lat, lon) to local (East, North) meters relative to a reference point.
    Uses a local flat earth approximation suitable for small distances (e.g., within a city).

    This transformation is critical for creating regression targets that are scaled
    appropriately (in meters) rather than degrees, which have different magnitudes
    depending on latitude.

    Args:
        lat_ref: Reference latitude in degrees (e.g., WLS baseline).
        lon_ref: Reference longitude in degrees (e.g., WLS baseline).
        lat: Target latitude in degrees (e.g., Ground Truth).
        lon: Target longitude in degrees (e.g., Ground Truth).

    Returns:
        d_east: Distance east in meters.
        d_north: Distance north in meters.
    """
    # Standard approximation: 1 degree latitude is approx 111320 meters
    DEG_TO_M = 111320.0

    # Ensure inputs are numpy arrays for vectorization
    lat_ref = np.array(lat_ref)
    lon_ref = np.array(lon_ref)
    lat = np.array(lat)
    lon = np.array(lon)

    # Calculate difference in degrees
    d_lat = lat - lat_ref
    d_lon = lon - lon_ref

    # Convert latitude difference to meters
    d_north = d_lat * DEG_TO_M

    # Convert longitude difference to meters
    # Scale by cosine of the reference latitude to account for meridian convergence
    scale_lon = np.cos(np.radians(lat_ref))
    d_east = d_lon * DEG_TO_M * scale_lon

    return d_east, d_north


def local_meters_to_wgs84(lat_ref, lon_ref, d_east, d_north):
    """
    Converts local (East, North) meters back to WGS84 (lat, lon) relative to a reference point.
    This is the inverse operation of wgs84_to_local_meters, used to reconstruct the final
    predicted path from the model's output residuals.

    Args:
        lat_ref: Reference latitude in degrees.
        lon_ref: Reference longitude in degrees.
        d_east: Distance east in meters (predicted residual).
        d_north: Distance north in meters (predicted residual).

    Returns:
        lat: Reconstructed latitude in degrees.
        lon: Reconstructed longitude in degrees.
    """
    DEG_TO_M = 111320.0

    lat_ref = np.array(lat_ref)
    lon_ref = np.array(lon_ref)
    d_east = np.array(d_east)
    d_north = np.array(d_north)

    # Convert North meters back to latitude degrees
    d_lat = d_north / DEG_TO_M
    lat = lat_ref + d_lat

    # Convert East meters back to longitude degrees
    scale_lon = np.cos(np.radians(lat_ref))
    # Clip scale to avoid division by zero near poles (though unlikely in this dataset)
    scale_lon = np.where(np.abs(scale_lon) < 1e-9, 1e-9, scale_lon)

    d_lon = d_east / (DEG_TO_M * scale_lon)
    lon = lon_ref + d_lon

    return lat, lon


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points on the earth (specified in decimal degrees).
    Used for computing the evaluation metric (mean of 50th and 95th percentile errors).

    Args:
        lat1, lon1: First point coordinates (scalar or array).
        lat2, lon2: Second point coordinates (scalar or array).

    Returns:
        Distance in meters.
    """
    R = 6371000.0  # Radius of Earth in meters

    # Convert degrees to radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)

    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )

    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return R * c
