import numpy as np
import pandas as pd
from library.utils import get_rotation_matrix, ecef_to_geodetic

# Constants
SPEED_OF_LIGHT = 299792458.0  # m/s


def compute_geometry(
    sat_pos_ecef: np.ndarray,
    user_pos_ecef: np.ndarray,
    raw_pr: np.ndarray,
    sat_clk_bias: np.ndarray,
    isrb: np.ndarray,
    iono_delay: np.ndarray,
    tropo_delay: np.ndarray,
):
    """
    Computes the geometry matrix G and residual vector r for a single epoch.

    Args:
        sat_pos_ecef (np.ndarray): (N, 3) Satellite positions in ECEF.
        user_pos_ecef (np.ndarray): (3,) User linearization point (WLS baseline) in ECEF.
        raw_pr (np.ndarray): (N,) Raw pseudoranges in meters.
        sat_clk_bias (np.ndarray): (N,) Satellite clock bias in meters (time * c).
        isrb (np.ndarray): (N,) Inter-signal range bias in meters.
        iono_delay (np.ndarray): (N,) Ionospheric delay in meters.
        tropo_delay (np.ndarray): (N,) Tropospheric delay in meters.

    Returns:
        G (np.ndarray): (N, 4) Design matrix [unit_vec_x, unit_vec_y, unit_vec_z, 1].
        r (np.ndarray): (N,) Pre-fit residuals.
    """
    # 1. Geometric Distance
    # Vector from user to satellite
    diff_vec = sat_pos_ecef - user_pos_ecef  # (N, 3)
    dist = np.linalg.norm(diff_vec, axis=1)  # (N,)

    # 2. Line of Sight Unit Vectors
    # Handle division by zero if dist is 0 (unlikely)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_vec = diff_vec / dist[:, np.newaxis]  # (N, 3)

    # 3. Corrected Pseudorange
    # rho_corr = raw_pr + sat_clk - isrb - iono - tropo
    # Note: Signs depend on convention.
    # Standard: Pr = Geom + c(dt_rx - dt_sat) + I + T + err
    # Corrected for geometric comparison:
    # Pr_corr = Pr + c*dt_sat - I - T - ISRB
    # Residual r = Pr_corr - Geom_Dist
    # r = c*dt_rx + err
    # Linearized: r = u * dx + c * dt_rx

    # Handle NaNs in corrections by replacing with 0
    sat_clk_bias = np.nan_to_num(sat_clk_bias)
    isrb = np.nan_to_num(isrb)
    iono_delay = np.nan_to_num(iono_delay)
    tropo_delay = np.nan_to_num(tropo_delay)

    pr_corr = raw_pr + sat_clk_bias - isrb - iono_delay - tropo_delay

    # 4. Residuals
    r = pr_corr - dist

    # 5. Design Matrix
    # G = [u_x, u_y, u_z, 1]
    N = len(raw_pr)
    G = np.hstack([u_vec, np.ones((N, 1))])

    return G, r


def solve_newton_step(G: np.ndarray, r: np.ndarray, weights: np.ndarray = None):
    """
    Solves the linearized GNSS equation for position and clock update.
    Delta x = (G^T W G)^-1 G^T W r

    Args:
        G (np.ndarray): (N, 4) Design matrix.
        r (np.ndarray): (N,) Residual vector.
        weights (np.ndarray, optional): (N,) Weights for WLS. Defaults to identity.

    Returns:
        delta_x (np.ndarray): (4,) Update vector [dx, dy, dz, dt].
        chi2 (float): Post-fit Chi-square statistic (weighted sum of squared residuals).
    """
    N = G.shape[0]
    if N < 4:
        # Not enough satellites to solve for 4 unknowns
        return np.zeros(4), np.nan

    if weights is None:
        weights = np.ones(N)

    # Construct W matrix (diagonal)
    W = np.diag(weights)

    # Normal Matrix
    GTG = G.T @ W @ G
    GTr = G.T @ W @ r

    try:
        # Add small regularization for stability
        GTG_reg = GTG + np.eye(4) * 1e-3
        delta_x = np.linalg.solve(GTG_reg, GTr)

        # Compute post-fit residuals
        # r_post = r - G @ delta_x
        r_post = r - G @ delta_x

        # Chi-square: sum(w * r^2) / dof
        # DOF = N - 4 (number of observations - number of unknowns)
        dof = N - 4
        if dof > 0:
            chi2 = np.sum(weights * (r_post**2)) / dof
        else:
            chi2 = 0.0

    except np.linalg.LinAlgError:
        delta_x = np.zeros(4)
        chi2 = np.nan

    return delta_x, chi2


def compute_dop(G: np.ndarray, weights: np.ndarray, user_pos_ecef: np.ndarray):
    """
    Computes Dilution of Precision (DOP) metrics in ENU frame.

    Args:
        G (np.ndarray): (N, 4) Design matrix (ECEF).
        weights (np.ndarray): (N,) Weights.
        user_pos_ecef (np.ndarray): (3,) User position for rotation.

    Returns:
        dict: {'GDOP', 'PDOP', 'HDOP', 'VDOP'}
    """
    N = G.shape[0]
    if N < 4:
        return {"GDOP": np.nan, "PDOP": np.nan, "HDOP": np.nan, "VDOP": np.nan}

    W = np.diag(weights)
    GTG = G.T @ W @ G

    try:
        # Covariance Matrix in ECEF (x, y, z, t)
        Q_ecef = np.linalg.inv(GTG)

        # We need to rotate the top-left 3x3 block (position) to ENU
        Q_xyz = Q_ecef[:3, :3]

        # Get Rotation Matrix
        lat, lon, _ = ecef_to_geodetic(
            user_pos_ecef[0], user_pos_ecef[1], user_pos_ecef[2]
        )
        R = get_rotation_matrix(np.radians(lat), np.radians(lon))  # (3, 3)

        # Q_enu = R @ Q_ecef @ R.T
        Q_enu = R @ Q_xyz @ R.T

        # Extract DOPs
        # Q_enu diagonal is [Var_E, Var_N, Var_U]
        # GDOP = sqrt(trace(Q_ecef)) (includes time)
        # PDOP = sqrt(Var_x + Var_y + Var_z)
        # HDOP = sqrt(Var_E + Var_N)
        # VDOP = sqrt(Var_U)

        gdop = np.sqrt(np.trace(Q_ecef))
        pdop = np.sqrt(np.trace(Q_xyz))
        hdop = np.sqrt(Q_enu[0, 0] + Q_enu[1, 1])
        vdop = np.sqrt(Q_enu[2, 2])

        return {"GDOP": gdop, "PDOP": pdop, "HDOP": hdop, "VDOP": vdop}

    except np.linalg.LinAlgError:
        return {"GDOP": np.nan, "PDOP": np.nan, "HDOP": np.nan, "VDOP": np.nan}


def process_gnss_data(df_gnss: pd.DataFrame):
    """
    High-level function to process a GNSS DataFrame and generate physics-based features.

    Args:
        df_gnss (pd.DataFrame): DataFrame containing raw GNSS logs.

    Returns:
        pd.DataFrame: Features indexed by 'utcTimeMillis'.
    """
    # Ensure sorted
    df_gnss = df_gnss.sort_values("utcTimeMillis")

    # Define columns needed
    req_cols = [
        "utcTimeMillis",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "RawPseudorangeMeters",
        "SvClockBiasMeters",
        "IsrbMeters",
        "IonosphericDelayMeters",
        "TroposphericDelayMeters",
        "Cn0DbHz",
        "SvElevationDegrees",
    ]

    # Filter valid rows (basic check)
    # We need valid WLS position to linearize around
    valid_mask = (
        df_gnss["WlsPositionXEcefMeters"].notna()
        & df_gnss["RawPseudorangeMeters"].notna()
    )
    df_clean = df_gnss.loc[valid_mask, req_cols].copy()

    results = []

    # Grouping by timestamp
    grouped = df_clean.groupby("utcTimeMillis")

    for time_millis, group in grouped:
        # Extract arrays
        sat_pos = group[
            ["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]
        ].values

        # User pos is constant for the epoch (WLS baseline)
        # Take the first valid one
        user_pos = (
            group[
                [
                    "WlsPositionXEcefMeters",
                    "WlsPositionYEcefMeters",
                    "WlsPositionZEcefMeters",
                ]
            ]
            .iloc[0]
            .values
        )

        raw_pr = group["RawPseudorangeMeters"].values
        sat_clk = group["SvClockBiasMeters"].values
        isrb = group["IsrbMeters"].values
        iono = group["IonosphericDelayMeters"].values
        tropo = group["TroposphericDelayMeters"].values
        cn0 = group["Cn0DbHz"].values
        el = group["SvElevationDegrees"].values

        # 1. Compute Geometry
        G, r = compute_geometry(sat_pos, user_pos, raw_pr, sat_clk, isrb, iono, tropo)

        # 2. Define Weights
        # Hypothesis A: Signal Strength (Cn0)
        # Use linear scale: 10^(Cn0/10)
        w_sig = 10 ** (cn0 / 10.0)

        # Hypothesis B: Elevation
        # sin(el)^2 is commonly used in RTKLIB
        el_rad = np.radians(el)
        w_el = np.square(np.sin(el_rad))

        # Hypothesis C: Mask (Quality)
        # Cn0 > 20
        w_mask = (cn0 > 20.0).astype(float)

        # 3. Solve Newton Steps
        # We want the position correction in ENU
        # Solve in ECEF -> Rotate

        # Hyp A
        dx_ecef_a, chi2_a = solve_newton_step(G, r, w_sig)

        # Hyp B
        dx_ecef_b, chi2_b = solve_newton_step(G, r, w_el)

        # Hyp C
        dx_ecef_c, chi2_c = solve_newton_step(G, r, w_mask)

        # Rotate corrections to ENU
        # Need rotation matrix at user_pos
        lat, lon, _ = ecef_to_geodetic(user_pos[0], user_pos[1], user_pos[2])
        R = get_rotation_matrix(np.radians(lat), np.radians(lon))

        # dx_enu = R @ dx_ecef[:3]
        dx_enu_a = R @ dx_ecef_a[:3]
        dx_enu_b = R @ dx_ecef_b[:3]
        dx_enu_c = R @ dx_ecef_c[:3]

        # 4. Compute DOP (using Hyp A weights as 'primary')
        dops = compute_dop(G, w_sig, user_pos)

        # 5. Collect Result
        res = {
            "utcTimeMillis": time_millis,
            "sat_count": len(raw_pr),
            # Hyp A
            "newton_E_sig": dx_enu_a[0],
            "newton_N_sig": dx_enu_a[1],
            "newton_U_sig": dx_enu_a[2],
            "chi2_sig": chi2_a,
            # Hyp B
            "newton_E_el": dx_enu_b[0],
            "newton_N_el": dx_enu_b[1],
            "newton_U_el": dx_enu_b[2],
            "chi2_el": chi2_b,
            # Hyp C
            "newton_E_mask": dx_enu_c[0],
            "newton_N_mask": dx_enu_c[1],
            "newton_U_mask": dx_enu_c[2],
            "chi2_mask": chi2_c,
            # DOPs
            "GDOP": dops["GDOP"],
            "PDOP": dops["PDOP"],
            "HDOP": dops["HDOP"],
            "VDOP": dops["VDOP"],
        }
        results.append(res)

    return pd.DataFrame(results)
