import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
from library.config import PHYSICS_LAMBDA, OPTIMIZER_LR, OPTIMIZER_EPOCHS, SEED
from library.utils import ecef_to_geodetic, enu_to_ecef

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


class TrajectoryModel(nn.Module):
    """
    PyTorch model representing the trajectory as learnable parameters.
    """

    def __init__(self, initial_positions):
        super().__init__()
        # Parameter is the trajectory (N, 3) in ECEF coordinates
        self.positions = nn.Parameter(
            torch.tensor(initial_positions, dtype=torch.float32)
        )

    def forward(self):
        return self.positions


def optimize_trip(trip_df):
    """
    Optimizes the trajectory for a single trip using Global L1 optimization.

    Args:
        trip_df (pd.DataFrame): Data for a single trip containing WLS positions,
                                predicted residuals, and Doppler velocities.

    Returns:
        pd.DataFrame: The trip dataframe with updated LatitudeDegrees and LongitudeDegrees.
    """
    # Ensure data is sorted by time
    trip_df = trip_df.sort_values("UnixTimeMillis").reset_index(drop=True)

    # 1. Prepare Anchor Points (WLS + Predicted Residuals)
    # Extract WLS ECEF positions
    wls_x = trip_df["WlsPositionXEcefMeters"].values
    wls_y = trip_df["WlsPositionYEcefMeters"].values
    wls_z = trip_df["WlsPositionZEcefMeters"].values

    # Convert WLS to Geodetic to serve as reference for ENU residuals
    # We use the WLS position itself as the local tangent plane origin for each point
    wls_lat, wls_lon, wls_alt = ecef_to_geodetic(wls_x, wls_y, wls_z)

    # Extract Predicted Residuals (ENU)
    # Fill NaNs with 0 (no correction) if any
    pred_e = trip_df["pred_e"].fillna(0.0).values
    pred_n = trip_df["pred_n"].fillna(0.0).values
    pred_u = np.zeros_like(pred_e)  # We assume 0 vertical correction

    # Convert ENU residuals to ECEF absolute positions (Anchors)
    anchor_x, anchor_y, anchor_z = enu_to_ecef(
        pred_e, pred_n, pred_u, wls_lat, wls_lon, wls_alt
    )

    anchors = np.stack([anchor_x, anchor_y, anchor_z], axis=1)

    # 2. Prepare Doppler Velocities
    # Extract Doppler velocities
    v_cols = ["v_doppler_x", "v_doppler_y", "v_doppler_z"]
    # Interpolate missing velocities to maintain continuity
    velocities = (
        trip_df[v_cols]
        .interpolate(method="linear", limit_direction="both")
        .fillna(0.0)
        .values
    )

    # 3. Calculate Time Deltas
    timestamps = trip_df["UnixTimeMillis"].values
    # dt[i] is time from i to i+1
    dt = np.diff(timestamps) / 1000.0

    # Convert to PyTorch Tensors
    t_anchors = torch.tensor(anchors, dtype=torch.float32)
    t_velocities = torch.tensor(velocities, dtype=torch.float32)  # Shape (N, 3)
    t_dt = torch.tensor(dt, dtype=torch.float32).unsqueeze(1)  # Shape (N-1, 1)

    # Initialize Model with Anchor positions
    model = TrajectoryModel(anchors)
    optimizer = optim.Adam(model.parameters(), lr=OPTIMIZER_LR)

    # Optimization Loop
    model.train()
    for _ in range(OPTIMIZER_EPOCHS):
        optimizer.zero_grad()

        curr_pos = model()

        # Term 1: Anchor Loss (L1 Norm)
        # || x_t - P_anchor_t ||_1
        # Robustness to outliers in the ML predictions
        loss_anchor = torch.mean(torch.abs(curr_pos - t_anchors))

        # Term 2: Physics Loss (L2 Norm)
        # || (x_t - x_{t-1}) - v_{t-1} * dt ||_2^2
        # Enforce kinematic consistency
        # We use velocities[:-1] corresponding to the interval start
        pos_diff = curr_pos[1:] - curr_pos[:-1]
        kinematic_step = t_velocities[:-1] * t_dt

        loss_physics = torch.mean(torch.sum((pos_diff - kinematic_step) ** 2, dim=1))

        # Composite Loss
        loss = loss_anchor + PHYSICS_LAMBDA * loss_physics

        loss.backward()
        optimizer.step()

    # Extract optimized positions
    opt_ecef = model.positions.detach().numpy()

    # Convert optimized ECEF back to Geodetic (Lat/Lon)
    opt_lat, opt_lon, _ = ecef_to_geodetic(
        opt_ecef[:, 0], opt_ecef[:, 1], opt_ecef[:, 2]
    )

    # Update DataFrame
    trip_df["LatitudeDegrees"] = opt_lat
    trip_df["LongitudeDegrees"] = opt_lon

    return trip_df


def optimize_trajectory(df):
    """
    Applies Global L1 Trajectory Optimization to the entire dataset.
    Groups data by tripId and optimizes each trip independently.

    Args:
        df (pd.DataFrame): Input dataframe containing:
                           - tripId, UnixTimeMillis
                           - WlsPosition[X/Y/Z]EcefMeters
                           - pred_e, pred_n (residuals)
                           - v_doppler_[x/y/z]

    Returns:
        pd.DataFrame: DataFrame with optimized LatitudeDegrees and LongitudeDegrees.
    """
    print(f"Optimizing trajectories for {df['tripId'].nunique()} trips...")

    trips = df["tripId"].unique()
    results = []

    for trip in trips:
        trip_df = df[df["tripId"] == trip].copy()

        # Skip empty trips
        if len(trip_df) == 0:
            continue

        try:
            opt_df = optimize_trip(trip_df)
            results.append(opt_df)
        except Exception as e:
            print(f"Error optimizing trip {trip}: {e}")
            # Fallback: use WLS converted to Lat/Lon if optimization fails
            wls_x = trip_df["WlsPositionXEcefMeters"].values
            wls_y = trip_df["WlsPositionYEcefMeters"].values
            wls_z = trip_df["WlsPositionZEcefMeters"].values
            lat, lon, _ = ecef_to_geodetic(wls_x, wls_y, wls_z)
            trip_df["LatitudeDegrees"] = lat
            trip_df["LongitudeDegrees"] = lon
            results.append(trip_df)

    if not results:
        return pd.DataFrame(
            columns=["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        )

    final_df = pd.concat(results, ignore_index=True)

    # Sort by trip and time to ensure order
    final_df = final_df.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)

    return final_df


def save_submission(df, output_path="./submission/submission.csv"):
    """
    Saves the optimized trajectory to a submission file.

    Args:
        df (pd.DataFrame): DataFrame containing 'tripId', 'UnixTimeMillis',
                           'LatitudeDegrees', 'LongitudeDegrees'.
        output_path (str): Path to save the CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Select required columns
    cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]

    # Check if columns exist
    if not all(col in df.columns for col in cols):
        print(
            f"Error: DataFrame missing required columns for submission. Available: {df.columns}"
        )
        return

    submission_df = df[cols]
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
