import os
import numpy as np
import pandas as pd
import torch
from library.config import (
    HUBER_DELTA,
    WEIGHT_TDCP,
    WEIGHT_DOPPLER,
    SEED,
    WORKING_DIR,
)
from library.utils import ecef_to_enu, enu_to_ecef, ecef_to_wgs84
from library.feature_builder import rotate_vector_ecef_to_enu

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


class TrajectoryGraph(torch.nn.Module):
    """
    PyTorch-based Factor Graph for Trajectory Optimization.
    Minimizes a cost function combining:
    1. Anchor Cost: Huber loss deviation from ML-predicted positions.
    2. Kinematic Cost: L2 loss deviation from TDCP/Doppler relative motions.
    """

    def __init__(self, anchors, deltas, weights_kin, huber_delta=HUBER_DELTA):
        """
        Args:
            anchors: (N, D) tensor of absolute position estimates (e.g., East, North).
            deltas: (N, D) tensor of kinematic changes (x_t - x_{t-1}).
            weights_kin: (N,) tensor of weights for the kinematic edges.
            huber_delta: Threshold for Huber loss.
        """
        super().__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.deltas = torch.tensor(deltas, dtype=torch.float32)
        self.weights_kin = torch.tensor(weights_kin, dtype=torch.float32)
        self.huber_delta = huber_delta

        # Parameter to optimize: The refined trajectory x
        # Initialize with the anchors (ML predictions)
        self.x = torch.nn.Parameter(self.anchors.clone())

    def forward(self):
        # 1. Anchor Cost (Unary Factors)
        # Robust Huber loss to handle outliers in ML predictions
        # Sum reduction to match the magnitude of kinematic terms
        anchor_loss = torch.nn.functional.huber_loss(
            self.x, self.anchors, reduction="sum", delta=self.huber_delta
        )

        # 2. Kinematic Cost (Binary Factors)
        # Enforce shape consistency: x_t - x_{t-1} should match delta_t
        # diff[t] corresponds to transition from t to t+1 in 0-indexed array?
        # Let's align with input: deltas[t] is transition to t from t-1.
        # So we compare (x[t] - x[t-1]) with deltas[t].
        # x[1:] - x[:-1] gives N-1 transitions.
        # deltas[1:] gives N-1 corresponding kinematic measurements.
        # weights_kin[1:] gives weights for these transitions.

        diff = self.x[1:] - self.x[:-1]
        kin_residuals = diff - self.deltas[1:]

        # Squared Euclidean Norm per transition
        kin_sq_errors = torch.sum(kin_residuals**2, dim=1)

        # Weighted sum L2 loss
        kin_loss = torch.sum(self.weights_kin[1:] * kin_sq_errors)

        return anchor_loss + kin_loss

    def optimize(self, max_iter=100, lr=1.0):
        """
        Run LBFGS optimization to minimize the graph cost.
        """
        # LBFGS is efficient for this type of convex-like trajectory smoothing
        optimizer = torch.optim.LBFGS(
            [self.x],
            lr=lr,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
            history_size=10,
        )

        def closure():
            optimizer.zero_grad()
            loss = self.forward()
            loss.backward()
            return loss

        optimizer.step(closure)

        return self.x.detach().numpy()


def process_trip(df):
    """
    Apply graph optimization to a single trip (drive-phone sequence).
    """
    # Ensure time order
    df = df.sort_values("UnixTimeMillis").reset_index(drop=True)

    # 1. Establish Local Tangent Plane (ENU)
    # Use the first WLS point as the reference origin
    wls_x = df["WlsPositionXEcefMeters"].values
    wls_y = df["WlsPositionYEcefMeters"].values
    wls_z = df["WlsPositionZEcefMeters"].values

    ref_x, ref_y, ref_z = wls_x[0], wls_y[0], wls_z[0]
    ref_lat, ref_lon, ref_alt = ecef_to_wgs84(ref_x, ref_y, ref_z)

    # Convert WLS baseline to ENU
    wls_e, wls_n, wls_u = ecef_to_enu(wls_x, wls_y, wls_z, ref_lat, ref_lon, ref_alt)

    # 2. Construct Anchors
    # Anchor = WLS + Predicted Residual
    # If predictions missing (e.g. in baseline mode), assume 0 residual
    pred_e = df["pred_E"].values if "pred_E" in df.columns else np.zeros_like(wls_e)
    pred_n = df["pred_N"].values if "pred_N" in df.columns else np.zeros_like(wls_n)

    anchors_e = wls_e + pred_e
    anchors_n = wls_n + pred_n

    # Stack for 2D optimization (East, North)
    anchors = np.column_stack((anchors_e, anchors_n))

    # 3. Construct Kinematics (Deltas)
    # Time difference in seconds
    t_diff = df["UnixTimeMillis"].diff().fillna(1000).values / 1000.0

    # Initialize ECEF deltas
    n_samples = len(df)
    delta_ecef_x = np.zeros(n_samples)
    delta_ecef_y = np.zeros(n_samples)
    delta_ecef_z = np.zeros(n_samples)
    weights = np.zeros(n_samples)

    # Check for kinematic columns
    has_tdcp = "TDCP_Disp_X" in df.columns
    has_doppler = "Doppler_Vel_X" in df.columns

    if has_tdcp:
        tdcp_valid = df["TDCP_Valid"].fillna(0).values.astype(bool)
        # Fill valid TDCP
        delta_ecef_x[tdcp_valid] = df.loc[tdcp_valid, "TDCP_Disp_X"].fillna(0).values
        delta_ecef_y[tdcp_valid] = df.loc[tdcp_valid, "TDCP_Disp_Y"].fillna(0).values
        delta_ecef_z[tdcp_valid] = df.loc[tdcp_valid, "TDCP_Disp_Z"].fillna(0).values
        weights[tdcp_valid] = WEIGHT_TDCP
    else:
        tdcp_valid = np.zeros(n_samples, dtype=bool)

    if has_doppler:
        # Fallback to Doppler where TDCP is invalid
        mask_dop = ~tdcp_valid
        dop_vx = df["Doppler_Vel_X"].fillna(0).values
        dop_vy = df["Doppler_Vel_Y"].fillna(0).values
        dop_vz = df["Doppler_Vel_Z"].fillna(0).values

        delta_ecef_x[mask_dop] = dop_vx[mask_dop] * t_diff[mask_dop]
        delta_ecef_y[mask_dop] = dop_vy[mask_dop] * t_diff[mask_dop]
        delta_ecef_z[mask_dop] = dop_vz[mask_dop] * t_diff[mask_dop]
        weights[mask_dop] = WEIGHT_DOPPLER
    elif not has_tdcp:
        # No kinematics available, zero weights (pure smoothing/anchor reliance)
        pass

    # First point has no history
    weights[0] = 0.0

    # Rotate ECEF Deltas to ENU
    # Note: We use the single reference point for rotation to stay in the local plane
    d_e, d_n, _ = rotate_vector_ecef_to_enu(
        delta_ecef_x, delta_ecef_y, delta_ecef_z, ref_lat, ref_lon
    )
    deltas = np.column_stack((d_e, d_n))

    # 4. Run Optimization
    graph = TrajectoryGraph(anchors, deltas, weights)
    opt_enu = graph.optimize()

    opt_e = opt_enu[:, 0]
    opt_n = opt_enu[:, 1]

    # 5. Convert Result back to Geodetic
    # ENU -> ECEF -> WGS84
    # We use the original WLS Up component as we only optimized horizontal
    opt_x, opt_y, opt_z = enu_to_ecef(opt_e, opt_n, wls_u, ref_lat, ref_lon, ref_alt)
    opt_lat, opt_lon, _ = ecef_to_wgs84(opt_x, opt_y, opt_z)

    df["LatitudeDegrees"] = opt_lat
    df["LongitudeDegrees"] = opt_lon

    return df


def optimize_dataframe(df, load_cached_data=True):
    """
    Apply graph optimization to all trips in the dataframe.
    """
    cache_path = os.path.join(WORKING_DIR, "optimized_predictions.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading optimized predictions from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Running Graph Optimization on dataframe...")

    results = []
    # Group by tripId to process independent trajectories
    trips = df["tripId"].unique()

    for trip_id in trips:
        group = df[df["tripId"] == trip_id].copy()
        try:
            opt_group = process_trip(group)
            results.append(opt_group)
        except Exception as e:
            print(f"Error optimizing trip {trip_id}: {e}")
            results.append(group)  # Fallback to original

    final_df = pd.concat(results, ignore_index=True)

    # Save to cache
    try:
        final_df.to_parquet(cache_path, index=False)
        print(f"Saved optimized predictions to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return final_df
