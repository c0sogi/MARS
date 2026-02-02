import numpy as np
import pandas as pd
from library.config import SPEED_OF_LIGHT


def calculate_los_vectors(rx_pos, sat_pos):
    """
    Compute Line-of-Sight (LOS) unit vectors and distances from receiver to satellites.

    Args:
        rx_pos: Receiver positions (N, 3) in ECEF meters.
        sat_pos: Satellite positions (N, 3) in ECEF meters.

    Returns:
        unit_vectors: (N, 3) array of unit vectors pointing from Receiver to Satellite.
        distances: (N,) array of Euclidean distances.
    """
    # Vector from Receiver to Satellite
    r_vec = sat_pos - rx_pos

    # Euclidean distance
    distances = np.linalg.norm(r_vec, axis=1)

    # Handle division by zero (though unlikely in GNSS)
    safe_distances = np.where(distances == 0, 1.0, distances)

    # Unit vectors
    unit_vectors = r_vec / safe_distances[:, np.newaxis]

    return unit_vectors, distances


def calculate_pseudorange_residuals(df):
    """
    Calculate pseudorange residuals relative to the WLS baseline.

    Residual = (RawPr + SatClk - Atm) - (GeometricDist + RxClk)

    We estimate RxClk per epoch as the median of the raw residuals to ensure
    the resulting residuals are zero-mean centered (robust to outliers).

    Args:
        df: DataFrame containing device_gnss.csv data.
            Must contain: RawPseudorangeMeters, SvClockBiasMeters, IsrbMeters,
                          IonosphericDelayMeters, TroposphericDelayMeters,
                          WlsPosition[X/Y/Z]EcefMeters, SvPosition[X/Y/Z]EcefMeters,
                          UnixTimeMillis.

    Returns:
        df: The input DataFrame with a new column 'PseudorangeResidualMeters'.
    """
    # 1. Calculate Corrected Pseudorange
    # CorrectedPr = Raw + SatClk - ISRB - Iono - Tropo
    # Note: We do not fillna(0) for SvClockBiasMeters as it is critical.
    # ISRB, Iono, Tropo can be 0 if missing without catastrophic failure.
    df["CorrectedPrMeters"] = (
        df["RawPseudorangeMeters"]
        + df["SvClockBiasMeters"]
        - df["IsrbMeters"].fillna(0)
        - df["IonosphericDelayMeters"].fillna(0)
        - df["TroposphericDelayMeters"].fillna(0)
    )

    # 2. Calculate Geometric Distance
    rx_pos = df[
        ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    ].values
    sat_pos = df[
        ["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]
    ].values

    # Use the LOS function
    _, distances = calculate_los_vectors(rx_pos, sat_pos)
    df["GeometricDistMeters"] = distances

    # 3. Calculate Raw Residuals (including Receiver Clock Bias)
    # RawRes = CorrectedPr - GeometricDist
    df["RawResidualMeters"] = df["CorrectedPrMeters"] - df["GeometricDistMeters"]

    # 4. Estimate Receiver Clock Bias per Epoch
    # We use the median of residuals at each timestamp as a robust estimator of c*dt_r
    # This assumes the position error is small compared to clock bias or zero-mean distributed.
    clock_bias_per_epoch = df.groupby("UnixTimeMillis")["RawResidualMeters"].transform(
        "median"
    )

    # 5. Calculate Final Post-Fit Residuals
    df["PseudorangeResidualMeters"] = df["RawResidualMeters"] - clock_bias_per_epoch

    return df


def calculate_carrier_phase_differences(df):
    """
    Compute Time-Differenced Carrier Phase (TDCP) observables between consecutive epochs.

    Filters based on AccumulatedDeltaRangeState to ensure continuous phase lock.

    Args:
        df: DataFrame containing device_gnss.csv data.
            Must contain: AccumulatedDeltaRangeMeters, AccumulatedDeltaRangeState,
                          Svid, SignalType, UnixTimeMillis,
                          SvPosition[X/Y/Z]EcefMeters.

    Returns:
        tdcp_df: DataFrame containing TDCP data.
            Columns: UnixTimeMillis, Svid, SignalType, TDCP_Meters,
                     SatDisp_X, SatDisp_Y, SatDisp_Z,
                     SvPositionXEcefMeters, SvPositionYEcefMeters, SvPositionZEcefMeters
    """
    # Filter valid ADR measurements
    # State bit 0 (1): Valid
    # State bit 1 (2): Reset (Must NOT be set)
    # State bit 2 (4): Cycle Slip (Must NOT be set)
    # We require Valid=1, Reset=0, CycleSlip=0.
    valid_mask = (
        ((df["AccumulatedDeltaRangeState"] & 1) == 1)
        & ((df["AccumulatedDeltaRangeState"] & 2) == 0)
        & ((df["AccumulatedDeltaRangeState"] & 4) == 0)
    )

    valid_df = df[valid_mask].copy()

    # Sort to ensure consecutive epochs are adjacent
    valid_df = valid_df.sort_values(by=["Svid", "SignalType", "UnixTimeMillis"])

    # Calculate differences
    # We group by Svid and SignalType to shift
    grouped = valid_df.groupby(["Svid", "SignalType"])

    valid_df["Prev_ADR"] = grouped["AccumulatedDeltaRangeMeters"].shift(1)
    valid_df["Prev_Time"] = grouped["UnixTimeMillis"].shift(1)
    valid_df["Prev_SvX"] = grouped["SvPositionXEcefMeters"].shift(1)
    valid_df["Prev_SvY"] = grouped["SvPositionYEcefMeters"].shift(1)
    valid_df["Prev_SvZ"] = grouped["SvPositionZEcefMeters"].shift(1)

    # Filter for consecutive epochs (gap <= 1000ms + tolerance)
    # Typical data is 1Hz. Allow small jitter.
    time_diff = valid_df["UnixTimeMillis"] - valid_df["Prev_Time"]
    consecutive_mask = (time_diff > 0) & (time_diff <= 1100)

    tdcp_df = valid_df[consecutive_mask].copy()

    # Calculate TDCP (Observed change in range + clock drift)
    tdcp_df["TDCP_Meters"] = (
        tdcp_df["AccumulatedDeltaRangeMeters"] - tdcp_df["Prev_ADR"]
    )

    # Calculate Satellite Displacement Vector (S_t - S_{t-1})
    tdcp_df["SatDisp_X"] = tdcp_df["SvPositionXEcefMeters"] - tdcp_df["Prev_SvX"]
    tdcp_df["SatDisp_Y"] = tdcp_df["SvPositionYEcefMeters"] - tdcp_df["Prev_SvY"]
    tdcp_df["SatDisp_Z"] = tdcp_df["SvPositionZEcefMeters"] - tdcp_df["Prev_SvZ"]

    # Return relevant columns
    cols = [
        "UnixTimeMillis",
        "Svid",
        "SignalType",
        "TDCP_Meters",
        "SatDisp_X",
        "SatDisp_Y",
        "SatDisp_Z",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
    ]

    return tdcp_df[cols]
