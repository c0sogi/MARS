import numpy as np
from library.config import WGS84_A, WGS84_F, WGS84_B

# Derived WGS84 Constants
WGS84_E2 = 1.0 - (WGS84_B**2 / WGS84_A**2)


def wgs84_to_ecef(lat, lon, alt):
    """
    Convert WGS84 Geodetic coordinates to ECEF (Earth-Centered, Earth-Fixed).

    Args:
        lat: Latitude in degrees (float or np.array)
        lon: Longitude in degrees (float or np.array)
        alt: Altitude in meters (float or np.array)

    Returns:
        x, y, z: ECEF coordinates in meters
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Radius of curvature in the prime vertical
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_wgs84(x, y, z):
    """
    Convert ECEF coordinates to WGS84 Geodetic.
    Uses Ferrari's solution for high precision conversion.

    Args:
        x, y, z: ECEF coordinates in meters (float or np.array)

    Returns:
        lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m)
    """
    # Distance from Z-axis
    p = np.sqrt(x**2 + y**2)

    # Longitude
    lon = np.arctan2(y, x)

    # Parameters for Ferrari's solution
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    e_prime_sq = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2

    num = z + e_prime_sq * WGS84_B * np.sin(theta) ** 3
    den = p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3

    lat = np.arctan2(num, den)

    # Calculate altitude
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Handle poles (where p is close to 0)
    # If p is very small, lat is +/- 90 degrees, and alt is abs(z) - b
    if np.isscalar(p):
        if p < 1.0:
            lat = np.sign(z) * np.pi / 2
            alt = np.abs(z) - WGS84_B
    else:
        # Array logic
        mask = p < 1.0
        if np.any(mask):
            lat[mask] = np.sign(z[mask]) * np.pi / 2
            alt[mask] = np.abs(z[mask]) - WGS84_B

    return np.rad2deg(lat), np.rad2deg(lon), alt


def ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt):
    """
    Convert ECEF coordinates to Local Tangent Plane (ENU - East, North, Up).

    Args:
        x, y, z: Target ECEF coordinates (float or np.array)
        ref_lat, ref_lon, ref_alt: Reference point WGS84 coordinates (scalars)

    Returns:
        e, n, u: East, North, Up coordinates in meters relative to reference
    """
    # Convert reference point to ECEF
    xr, yr, zr = wgs84_to_ecef(ref_lat, ref_lon, ref_alt)

    dx = x - xr
    dy = y - yr
    dz = z - zr

    phi = np.deg2rad(ref_lat)
    lam = np.deg2rad(ref_lon)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Rotation matrix R applied to delta vector
    # R = [ -sin_lam,           cos_lam,          0      ]
    #     [ -sin_phi*cos_lam,  -sin_phi*sin_lam,  cos_phi]
    #     [  cos_phi*cos_lam,   cos_phi*sin_lam,  sin_phi]

    e = -sin_lam * dx + cos_lam * dy
    n = -sin_phi * cos_lam * dx - sin_phi * sin_lam * dy + cos_phi * dz
    u = cos_phi * cos_lam * dx + cos_phi * sin_lam * dy + sin_phi * dz

    return e, n, u


def enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt):
    """
    Convert ENU coordinates back to ECEF.

    Args:
        e, n, u: ENU coordinates in meters (float or np.array)
        ref_lat, ref_lon, ref_alt: Reference point WGS84 coordinates (scalars)

    Returns:
        x, y, z: ECEF coordinates in meters
    """
    xr, yr, zr = wgs84_to_ecef(ref_lat, ref_lon, ref_alt)

    phi = np.deg2rad(ref_lat)
    lam = np.deg2rad(ref_lon)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Inverse Rotation (Transpose of R) applied to ENU vector
    # R.T = [ -sin_lam,  -sin_phi*cos_lam,  cos_phi*cos_lam ]
    #       [  cos_lam,  -sin_phi*sin_lam,  cos_phi*sin_lam ]
    #       [  0,         cos_phi,          sin_phi         ]

    dx = -sin_lam * e - sin_phi * cos_lam * n + cos_phi * cos_lam * u
    dy = cos_lam * e - sin_phi * sin_lam * n + cos_phi * sin_lam * u
    dz = cos_phi * n + sin_phi * u

    x = xr + dx
    y = yr + dy
    z = zr + dz

    return x, y, z
