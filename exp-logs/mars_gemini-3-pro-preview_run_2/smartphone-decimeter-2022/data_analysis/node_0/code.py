import os
import numpy as np
import pandas as pd
import warnings
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
RANDOM_STATE = 42
SAMPLE_TRIPS_COUNT = 3

# Set random seeds
np.random.seed(RANDOM_STATE)


def ecef_to_lla(x, y, z):
    # WGS84 ellipsoid constants
    a = 6378137.0
    e = 8.1819190842622e-2

    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(
        (z + ep**2 * b * np.sin(th) ** 3), (p - e**2 * a * np.cos(th) ** 3)
    )

    # Convert to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def analyze_tabular_column(series, name):
    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe()
        q1 = desc["25%"]
        q3 = desc["75%"]
        iqr = q3 - q1
        outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()
        print(
            f"  {name}: Mean={desc['mean']:.4f}, Std={desc['std']:.4f}, Min={desc['min']:.4f}, Max={desc['max']:.4f}, Outliers={outliers}"
        )
    else:
        n_unique = series.nunique()
        print(f"  {name}: Cardinality={n_unique}")
        if n_unique > 50:
            print(f"    [FLAG] High cardinality (>50)")

        # Check rare labels
        counts = series.value_counts(normalize=True)
        rare = counts[counts < 0.01]
        if not rare.empty:
            print(f"    [FLAG] Rare labels detected (<1%): {len(rare)} labels")


def main():
    print("==================================================")
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("==================================================")

    # ---------------------------------------------------------
    # 1. Target Variable Analysis
    # ---------------------------------------------------------
    print("\nTARGET VARIABLE ANALYSIS")
    print("------------------------")

    if not os.path.exists(TRAIN_METADATA_PATH):
        print(f"Error: Metadata file not found at {TRAIN_METADATA_PATH}")
        return

    df_meta = pd.read_csv(TRAIN_METADATA_PATH)

    targets = ["LatitudeDegrees", "LongitudeDegrees"]

    for target in targets:
        data = df_meta[target]
        print(f"Variable: {target}")
        print(f"  Mean: {data.mean():.4f}")
        print(f"  Std : {data.std():.4f}")
        print(f"  Min : {data.min():.4f}")
        print(f"  Max : {data.max():.4f}")

        s = skew(data)
        k = kurtosis(data)
        print(f"  Skewness: {s:.4f}")
        print(f"  Kurtosis: {k:.4f}")
        print("")

    # ---------------------------------------------------------
    # 2. Input Data Analysis (Tabular)
    # ---------------------------------------------------------
    print("INPUT DATA ANALYSIS (TABULAR)")
    print("-----------------------------")

    # Sample trips
    unique_trips = df_meta["tripId"].unique()
    if len(unique_trips) > SAMPLE_TRIPS_COUNT:
        sampled_trips = np.random.choice(
            unique_trips, SAMPLE_TRIPS_COUNT, replace=False
        )
    else:
        sampled_trips = unique_trips

    print(f"Analyzing a sample of {len(sampled_trips)} trips: {sampled_trips}")

    gnss_data_list = []
    imu_data_list = []

    for trip_id in sampled_trips:
        trip_meta = df_meta[df_meta["tripId"] == trip_id].iloc[0]

        # Construct paths
        gnss_path = os.path.join(INPUT_DIR, trip_meta["gnss_path"])
        imu_path = os.path.join(INPUT_DIR, trip_meta["imu_path"])

        if os.path.exists(gnss_path):
            gnss_df = pd.read_csv(gnss_path)
            gnss_df["tripId"] = trip_id
            gnss_data_list.append(gnss_df)

        if os.path.exists(imu_path):
            imu_df = pd.read_csv(imu_path)
            imu_df["tripId"] = trip_id
            imu_data_list.append(imu_df)

    if not gnss_data_list:
        print("No GNSS data found for sampled trips.")
        return

    df_gnss = pd.concat(gnss_data_list, ignore_index=True)
    df_imu = (
        pd.concat(imu_data_list, ignore_index=True) if imu_data_list else pd.DataFrame()
    )

    print(f"\nGNSS Data Shape: {df_gnss.shape}")
    print("GNSS Numerical Features:")
    gnss_nums = [
        "Cn0DbHz",
        "RawPseudorangeMeters",
        "RawPseudorangeUncertaintyMeters",
        "ReceivedSvTimeUncertaintyNanos",
    ]
    for col in gnss_nums:
        if col in df_gnss.columns:
            analyze_tabular_column(df_gnss[col], col)

    print("\nGNSS Categorical Features:")
    gnss_cats = ["ConstellationType", "SignalType", "Svid"]
    for col in gnss_cats:
        if col in df_gnss.columns:
            analyze_tabular_column(df_gnss[col], col)

    print("\nGNSS Missing Values:")
    missing = df_gnss.isnull().mean() * 100
    missing = missing[missing > 0]
    if missing.empty:
        print("  No missing values.")
    else:
        for col, pct in missing.items():
            if pct > 1.0:  # Only show if > 1%
                print(f"  {col}: {pct:.2f}%")

    if not df_imu.empty:
        print(f"\nIMU Data Shape: {df_imu.shape}")
        print("IMU Numerical Features (Sample):")
        imu_nums = ["MeasurementX", "MeasurementY", "MeasurementZ"]
        for col in imu_nums:
            if col in df_imu.columns:
                analyze_tabular_column(df_imu[col], col)

    # ---------------------------------------------------------
    # 3. Feature/Signal Relationships
    # ---------------------------------------------------------
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("----------------------------")

    # We will aggregate GNSS data by epoch (utcTimeMillis) to align with Ground Truth
    # Features to aggregate
    agg_funcs = {
        "Svid": "count",
        "Cn0DbHz": "mean",
        "RawPseudorangeUncertaintyMeters": "mean",
        "WlsPositionXEcefMeters": "first",  # Baseline position
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
    }

    # Filter only columns that exist
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in df_gnss.columns}

    # Group by Trip and Time
    df_gnss_grouped = (
        df_gnss.groupby(["tripId", "utcTimeMillis"]).agg(agg_funcs).reset_index()
    )
    df_gnss_grouped.rename(
        columns={
            "Svid": "SatelliteCount",
            "Cn0DbHz": "MeanCn0",
            "RawPseudorangeUncertaintyMeters": "MeanUncertainty",
        },
        inplace=True,
    )

    # Merge with Ground Truth (Metadata)
    # Metadata has UnixTimeMillis. GNSS has utcTimeMillis. They should be comparable.
    # We need to join on tripId and Time.

    # Prepare metadata for merge
    df_meta_subset = df_meta[df_meta["tripId"].isin(sampled_trips)].copy()

    # Merge
    # Note: Timestamps might not match exactly ms to ms. We use nearest merge or exact if aligned.
    # The dataset description says "Reference locations at expected timestamps".
    # Let's try exact merge first.
    merged_df = pd.merge(
        df_gnss_grouped,
        df_meta_subset[
            ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        ],
        left_on=["tripId", "utcTimeMillis"],
        right_on=["tripId", "UnixTimeMillis"],
        how="inner",
    )

    if merged_df.empty:
        print(
            "  Warning: No exact timestamp matches found between GNSS and Ground Truth. Skipping correlation analysis."
        )
    else:
        print(f"  Matched {len(merged_df)} epochs for analysis.")

        # Calculate Baseline Error
        # Convert WLS ECEF to Lat/Lon
        if "WlsPositionXEcefMeters" in merged_df.columns:
            wls_lat, wls_lon = ecef_to_lla(
                merged_df["WlsPositionXEcefMeters"].values,
                merged_df["WlsPositionYEcefMeters"].values,
                merged_df["WlsPositionZEcefMeters"].values,
            )

            merged_df["BaselineErrorMeters"] = haversine_distance(
                merged_df["LatitudeDegrees"],
                merged_df["LongitudeDegrees"],
                wls_lat,
                wls_lon,
            )

            print(
                f"  Baseline WLS Error Mean: {merged_df['BaselineErrorMeters'].mean():.4f} m"
            )

            # Correlation
            analysis_cols = [
                "SatelliteCount",
                "MeanCn0",
                "MeanUncertainty",
                "BaselineErrorMeters",
            ]
            # Filter cols that exist
            analysis_cols = [c for c in analysis_cols if c in merged_df.columns]

            corr_matrix = merged_df[analysis_cols].corr(method="spearman")
            print("\n  Spearman Correlation with Baseline Error:")
            if "BaselineErrorMeters" in corr_matrix:
                print(
                    corr_matrix["BaselineErrorMeters"]
                    .drop("BaselineErrorMeters")
                    .to_string()
                )

            # Feature Importance (Random Forest)
            print("\n  Feature Importance (Predicting Baseline Error):")
            features = [c for c in analysis_cols if c != "BaselineErrorMeters"]
            X = merged_df[features]
            y = merged_df["BaselineErrorMeters"]

            # Impute if any NaNs (though aggregation usually handles it, WLS might be null)
            imputer = SimpleImputer(strategy="mean")
            X_imputed = imputer.fit_transform(X)

            # Drop rows where target is NaN
            mask = ~np.isnan(y)
            X_final = X_imputed[mask]
            y_final = y[mask]

            if len(y_final) > 10:
                rf = RandomForestRegressor(
                    n_estimators=50, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1
                )
                rf.fit(X_final, y_final)

                importances = pd.Series(
                    rf.feature_importances_, index=features
                ).sort_values(ascending=False)
                print(importances.to_string())
            else:
                print("  Not enough data points for RF training.")


if __name__ == "__main__":
    main()
