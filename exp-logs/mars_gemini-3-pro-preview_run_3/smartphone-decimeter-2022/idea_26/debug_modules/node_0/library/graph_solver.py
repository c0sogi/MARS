import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
from library.config import WORKING_DIR, GRAPH_PARAMS
from library.utils import enu_to_ecef, ecef_to_geodetic, geodetic_to_ecef


def optimize_trajectory(
    dataset_df: pd.DataFrame, kinematics_df: pd.DataFrame, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Applies State-Uncertainty Weighted Graph Optimization to fuse ML predictions with Kinematics.

    Args:
        dataset_df: DataFrame containing 'tripId', 'UnixTimeMillis', 'pred_E', 'pred_N',
                    'WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters',
                    'BiasUncertaintyNanos'.
        kinematics_df: DataFrame containing 'tripId', 'UnixTimeMillis', 'kin_x', 'kin_y', 'kin_z', 'kin_weight'.
        load_cached_data: If True, attempts to load result from parquet cache.

    Returns:
        DataFrame with optimized 'LatitudeDegrees' and 'LongitudeDegrees'.
    """
    cache_path = os.path.join(WORKING_DIR, "optimized_trajectory.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading optimized trajectory from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Starting Graph Optimization...")

    # Ensure inputs are sorted
    dataset_df = dataset_df.sort_values(["tripId", "UnixTimeMillis"]).reset_index(
        drop=True
    )

    # Prepare Kinematics
    # Kinematics defines the edge from t-1 to t. We merge on current time t.
    kin_cols = ["tripId", "UnixTimeMillis", "kin_x", "kin_y", "kin_z", "kin_weight"]
    # Ensure kinematics_df has unique keys
    kinematics_df = kinematics_df[kin_cols].drop_duplicates(
        subset=["tripId", "UnixTimeMillis"]
    )

    # Merge Data
    # Left join because we must have an anchor for every timestamp in the test set,
    # but might not have kinematics for every step.
    merged_df = pd.merge(
        dataset_df, kinematics_df, on=["tripId", "UnixTimeMillis"], how="left"
    )

    # Fill missing kinematics with 0 weight (no constraint)
    merged_df["kin_weight"] = merged_df["kin_weight"].fillna(0.0)
    merged_df["kin_x"] = merged_df["kin_x"].fillna(0.0)
    merged_df["kin_y"] = merged_df["kin_y"].fillna(0.0)
    merged_df["kin_z"] = merged_df["kin_z"].fillna(0.0)

    # Fill missing BiasUncertainty with a high value (low confidence) if any
    if "BiasUncertaintyNanos" not in merged_df.columns:
        # Fallback if column missing (e.g. minimal test set)
        merged_df["BiasUncertaintyNanos"] = 1.0e9
    merged_df["BiasUncertaintyNanos"] = merged_df["BiasUncertaintyNanos"].fillna(1.0e9)

    results = []

    # Process per trip
    for trip_id, group in tqdm(merged_df.groupby("tripId"), desc="Optimizing Trips"):
        group = group.reset_index(drop=True)
        n = len(group)

        if n == 0:
            continue

        # 1. Compute Anchor Positions (P_ML) in ECEF
        # We need to convert pred_E, pred_N (ENU residuals) to ECEF offsets

        # Get WLS Geodetic for ENU conversion reference
        wls_x = group["WlsPositionXEcefMeters"].values
        wls_y = group["WlsPositionYEcefMeters"].values
        wls_z = group["WlsPositionZEcefMeters"].values

        lat_wls, lon_wls, alt_wls = ecef_to_geodetic(wls_x, wls_y, wls_z)

        # Predicted residuals (fill NaN with 0)
        dE = group["pred_E"].fillna(0.0).values
        dN = group["pred_N"].fillna(0.0).values
        dU = np.zeros(n)  # We don't predict Up, assume 0 offset from WLS altitude

        # Convert ENU residuals to ECEF residuals
        # Note: enu_to_ecef returns absolute coordinates if we pass the reference as origin
        # Here we want the absolute ML position: P_ML = P_WLS + Rot * d_ENU
        # Our utils.enu_to_ecef does: x = x0 + dx. So passing P_WLS as ref gives P_ML.
        ml_x, ml_y, ml_z = enu_to_ecef(dE, dN, dU, lat_wls, lon_wls, alt_wls)

        # 2. Compute Anchor Weights
        # Weight ~ 1 / Uncertainty.
        # Add epsilon to avoid division by zero.
        # Scale by base parameter.
        uncertainty = group["BiasUncertaintyNanos"].values
        # Clip uncertainty to reasonable bounds to prevent exploding weights
        uncertainty = np.clip(uncertainty, 1.0, 1.0e12)
        w_anc = GRAPH_PARAMS["anchor_weight_base"] / uncertainty

        # 3. Construct Sparse Linear System
        # Variables: x_0, ..., x_{n-1} (for each dimension X, Y, Z independently)
        # Equations:
        #   Anchor: w_anc[t] * x[t] = w_anc[t] * ml_pos[t]
        #   Edge:   w_kin[t] * (x[t] - x[t-1]) = w_kin[t] * kin_disp[t]

        # Total equations = n (anchors) + n-1 (edges)
        # However, we can just loop t from 1 to n-1 for edges.

        # Kinematic data
        k_x = group["kin_x"].values
        k_y = group["kin_y"].values
        k_z = group["kin_z"].values
        w_kin = group["kin_weight"].values

        # Build Matrix A data
        # We use lil_matrix or build COO data arrays for efficiency
        rows = []
        cols = []
        data = []

        # RHS vectors
        bx = []
        by = []
        bz = []

        row_idx = 0

        # Anchor Constraints (t=0 to n-1)
        for t in range(n):
            weight = w_anc[t]

            # Equation: w * x[t] = w * ml[t]
            rows.append(row_idx)
            cols.append(t)
            data.append(weight)

            bx.append(weight * ml_x[t])
            by.append(weight * ml_y[t])
            bz.append(weight * ml_z[t])

            row_idx += 1

        # Edge Constraints (t=1 to n-1)
        for t in range(1, n):
            weight = w_kin[t]

            if weight > 0:
                # Equation: w * (x[t] - x[t-1]) = w * delta[t]
                # w * x[t] - w * x[t-1] = w * delta

                # Term x[t]
                rows.append(row_idx)
                cols.append(t)
                data.append(weight)

                # Term x[t-1]
                rows.append(row_idx)
                cols.append(t - 1)
                data.append(-weight)

                bx.append(weight * k_x[t])
                by.append(weight * k_y[t])
                bz.append(weight * k_z[t])

                row_idx += 1

        # Construct Sparse Matrix A
        A = sparse.coo_matrix((data, (rows, cols)), shape=(row_idx, n)).tocsr()

        bx = np.array(bx)
        by = np.array(by)
        bz = np.array(bz)

        # Solve Least Squares: (A^T A) x = A^T b
        # scipy.sparse.linalg.spsolve solves Ax=b for square A, but here A is rectangular (overdetermined).
        # We solve the normal equations: (A.T @ A) x = A.T @ b
        # This is efficient for sparse matrices.

        AtA = A.T @ A
        Atbx = A.T @ bx
        Atby = A.T @ by
        Atbz = A.T @ bz

        # Solve
        # use spsolve (direct solver) as the system is tridiagonal-like and sparse
        opt_x = spsolve(AtA, Atbx)
        opt_y = spsolve(AtA, Atby)
        opt_z = spsolve(AtA, Atbz)

        # Convert Optimized ECEF to Geodetic
        opt_lat, opt_lon, _ = ecef_to_geodetic(opt_x, opt_y, opt_z)

        # Store results
        trip_res = pd.DataFrame(
            {
                "tripId": group["tripId"],
                "UnixTimeMillis": group["UnixTimeMillis"],
                "LatitudeDegrees": opt_lat,
                "LongitudeDegrees": opt_lon,
            }
        )

        results.append(trip_res)

    final_df = pd.concat(results, ignore_index=True)

    # Sort to ensure alignment with sample submission if needed
    final_df = final_df.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)

    print(f"Saving optimized trajectory to cache: {cache_path}")
    final_df.to_parquet(cache_path, index=False)

    return final_df
