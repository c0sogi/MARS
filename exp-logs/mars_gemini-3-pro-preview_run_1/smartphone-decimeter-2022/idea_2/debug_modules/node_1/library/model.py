import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from library.config import Config


# -------------------------------------------------------------------------
# 1. Helper Functions (Coordinate Transforms)
# -------------------------------------------------------------------------
def ecef_to_lla(x, y, z):
    # WGS84 ellipsoid constants
    a = 6378137.0
    e = 8.1819190842622e-2

    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep**2 * b * np.sin(th) ** 3, p - e**2 * a * np.cos(th) ** 3)
    N = a / np.sqrt(1 - e**2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    return np.degrees(lat), np.degrees(lon), alt


# -------------------------------------------------------------------------
# 2. TCN Model Definition
# -------------------------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SensorFusionTCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(SensorFusionTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 2)  # Output: Delta Lat, Delta Lon

    def forward(self, x):
        # x shape: (Batch, Features, Seq_Len)
        y = self.network(x)
        # We only care about the prediction at the last time step for the window
        y_last = y[:, :, -1]
        return self.linear(y_last)


# -------------------------------------------------------------------------
# 3. Data Processing
# -------------------------------------------------------------------------
def process_drive_data(drive_id, phone_name, df_meta_subset, load_cached_data=True):
    """
    Processes a single drive's data: aligns GNSS, IMU, and GT (if available).
    Returns a DataFrame with features and targets (if train).
    """
    cache_file = os.path.join(Config.WORKING_DIR, f"{drive_id}_{phone_name}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    # 1. Identify file paths from metadata
    if df_meta_subset.empty:
        return pd.DataFrame()

    row = df_meta_subset.iloc[0]
    gnss_path = os.path.join(Config.INPUT_DIR, row["gnss_path"])
    imu_path = os.path.join(Config.INPUT_DIR, row["imu_path"])

    # 2. Load GNSS
    try:
        gnss_df = pd.read_csv(gnss_path)
    except FileNotFoundError:
        return pd.DataFrame()

    # 3. Load IMU
    try:
        imu_df = pd.read_csv(imu_path)
    except FileNotFoundError:
        imu_df = pd.DataFrame()

    # 4. Process GNSS (Aggregate by epoch)
    gnss_agg = (
        gnss_df.groupby("utcTimeMillis")
        .agg(
            {
                "Cn0DbHz": "mean",
                "SvElevationDegrees": "mean",
                "RawPseudorangeUncertaintyMeters": "mean",
                "Svid": "count",
                "WlsPositionXEcefMeters": "first",
                "WlsPositionYEcefMeters": "first",
                "WlsPositionZEcefMeters": "first",
            }
        )
        .reset_index()
    )

    gnss_agg.rename(
        columns={
            "Cn0DbHz": Config.FEAT_GNSS_CN0,
            "SvElevationDegrees": Config.FEAT_GNSS_ELEV,
            "RawPseudorangeUncertaintyMeters": Config.FEAT_GNSS_UNCERTAINTY,
            "Svid": Config.FEAT_GNSS_SAT_COUNT,
        },
        inplace=True,
    )

    # Convert WLS ECEF to LLA
    wls_lat, wls_lon, wls_alt = ecef_to_lla(
        gnss_agg["WlsPositionXEcefMeters"].values,
        gnss_agg["WlsPositionYEcefMeters"].values,
        gnss_agg["WlsPositionZEcefMeters"].values,
    )
    gnss_agg["WlsLat"] = wls_lat
    gnss_agg["WlsLon"] = wls_lon
    gnss_agg[Config.FEAT_WLS_ALT] = wls_alt

    # Calculate WLS Velocity (Delta)
    gnss_agg[Config.FEAT_WLS_LAT_DELTA] = gnss_agg["WlsLat"].diff().fillna(0)
    gnss_agg[Config.FEAT_WLS_LON_DELTA] = gnss_agg["WlsLon"].diff().fillna(0)

    # 5. Process IMU (Align to GNSS timestamps)
    if not imu_df.empty:
        # Calculate Magnitude
        accel = imu_df[imu_df["MessageType"] == "UncalAccel"].copy()
        if not accel.empty:
            accel["mag"] = np.sqrt(
                accel["MeasurementX"] ** 2
                + accel["MeasurementY"] ** 2
                + accel["MeasurementZ"] ** 2
            )
            accel["TimeBin"] = (accel["utcTimeMillis"] // 1000) * 1000
            accel_agg = (
                accel.groupby("TimeBin")["mag"].agg(["mean", "std"]).reset_index()
            )
            accel_agg.rename(
                columns={
                    "mean": Config.FEAT_IMU_ACCEL_MEAN,
                    "std": Config.FEAT_IMU_ACCEL_STD,
                },
                inplace=True,
            )
        else:
            accel_agg = pd.DataFrame(
                columns=[
                    "TimeBin",
                    Config.FEAT_IMU_ACCEL_MEAN,
                    Config.FEAT_IMU_ACCEL_STD,
                ]
            )

        gyro = imu_df[imu_df["MessageType"] == "UncalGyro"].copy()
        if not gyro.empty:
            gyro["TimeBin"] = (gyro["utcTimeMillis"] // 1000) * 1000
            gyro_agg = gyro.groupby("TimeBin")["MeasurementZ"].mean().reset_index()
            gyro_agg.rename(
                columns={"MeasurementZ": Config.FEAT_IMU_GYRO_Z_MEAN}, inplace=True
            )
        else:
            gyro_agg = pd.DataFrame(columns=["TimeBin", Config.FEAT_IMU_GYRO_Z_MEAN])

        # Merge IMU features
        gnss_agg["TimeBin"] = (gnss_agg["utcTimeMillis"] // 1000) * 1000

        gnss_agg = pd.merge(gnss_agg, accel_agg, on="TimeBin", how="left")
        gnss_agg = pd.merge(gnss_agg, gyro_agg, on="TimeBin", how="left")

        # Fill missing IMU data
        gnss_agg[Config.FEAT_IMU_ACCEL_MEAN] = gnss_agg[
            Config.FEAT_IMU_ACCEL_MEAN
        ].fillna(9.8)
        gnss_agg[Config.FEAT_IMU_ACCEL_STD] = gnss_agg[
            Config.FEAT_IMU_ACCEL_STD
        ].fillna(0)
        gnss_agg[Config.FEAT_IMU_GYRO_Z_MEAN] = gnss_agg[
            Config.FEAT_IMU_GYRO_Z_MEAN
        ].fillna(0)

        gnss_agg.drop(columns=["TimeBin"], inplace=True)
    else:
        gnss_agg[Config.FEAT_IMU_ACCEL_MEAN] = 9.8
        gnss_agg[Config.FEAT_IMU_ACCEL_STD] = 0.0
        gnss_agg[Config.FEAT_IMU_GYRO_Z_MEAN] = 0.0

    # 6. Merge with Metadata (Targets or TripId)
    df_meta_subset = df_meta_subset.rename(columns={"UnixTimeMillis": "utcTimeMillis"})

    if "LatitudeDegrees" in df_meta_subset.columns:
        # Train/Val case
        merged_df = pd.merge(
            gnss_agg,
            df_meta_subset[["utcTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
            on="utcTimeMillis",
            how="left",
        )
        merged_df[Config.TARGET_LAT_RES] = (
            merged_df["LatitudeDegrees"] - merged_df["WlsLat"]
        )
        merged_df[Config.TARGET_LON_RES] = (
            merged_df["LongitudeDegrees"] - merged_df["WlsLon"]
        )
    else:
        # Test case
        merged_df = pd.merge(
            gnss_agg,
            df_meta_subset[["utcTimeMillis", "tripId"]],
            on="utcTimeMillis",
            how="left",
        )
        merged_df[Config.TARGET_LAT_RES] = np.nan
        merged_df[Config.TARGET_LON_RES] = np.nan

    merged_df["drive_id"] = drive_id
    merged_df["phone_name"] = phone_name

    merged_df.to_parquet(cache_file)
    return merged_df


class WindowedDataset(Dataset):
    def __init__(self, data_df, window_size, mode="train"):
        self.data = data_df
        self.window_size = window_size
        self.mode = mode
        self.indices = []

        groups = data_df.groupby(["drive_id", "phone_name"])

        for _, group in groups:
            group_indices = group.index.values

            if mode == "test":
                # Valid targets are those with a tripId
                valid_mask = ~group["tripId"].isna()
            else:
                valid_mask = ~group[Config.TARGET_LAT_RES].isna()

            for i in range(len(group)):
                if valid_mask.iloc[i]:
                    if i >= window_size - 1:
                        self.indices.append(group_indices[i])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        df_idx = self.indices[idx]
        start_idx = df_idx - self.window_size + 1
        end_idx = df_idx + 1

        window_df = self.data.iloc[start_idx:end_idx]
        features = (
            window_df[Config.INPUT_FEATURES].values.astype(np.float32).transpose(1, 0)
        )

        if self.mode != "test":
            target = np.array(
                [
                    self.data.iloc[df_idx][Config.TARGET_LAT_RES],
                    self.data.iloc[df_idx][Config.TARGET_LON_RES],
                ],
                dtype=np.float32,
            )
            return features, target
        else:
            wls = np.array(
                [self.data.iloc[df_idx]["WlsLat"], self.data.iloc[df_idx]["WlsLon"]],
                dtype=np.float64,
            )
            return features, wls


# -------------------------------------------------------------------------
# 4. Training & Evaluation
# -------------------------------------------------------------------------
def train_model(train_meta_path, val_meta_path):
    print("Loading metadata...")
    train_meta = pd.read_csv(train_meta_path)
    val_meta = pd.read_csv(val_meta_path)

    if Config.DEBUG:
        print(f"DEBUG MODE: Sampling {Config.DEBUG_DRIVE_COUNT} drives.")
        drives = train_meta["drive_id"].unique()[: Config.DEBUG_DRIVE_COUNT]
        train_meta = train_meta[train_meta["drive_id"].isin(drives)]
        val_meta = val_meta[val_meta["drive_id"].isin(drives)]

    print("Processing Training Data...")
    train_dfs = []
    for (drive_id, phone_name), group in train_meta.groupby(["drive_id", "phone_name"]):
        df = process_drive_data(drive_id, phone_name, group, load_cached_data=True)
        if not df.empty:
            train_dfs.append(df)

    if not train_dfs:
        print("No training data found.")
        return None

    full_train_df = pd.concat(train_dfs, ignore_index=True)
    full_train_df.sort_values(["drive_id", "phone_name", "utcTimeMillis"], inplace=True)
    full_train_df.reset_index(drop=True, inplace=True)

    print("Processing Validation Data...")
    val_dfs = []
    for (drive_id, phone_name), group in val_meta.groupby(["drive_id", "phone_name"]):
        df = process_drive_data(drive_id, phone_name, group, load_cached_data=True)
        if not df.empty:
            val_dfs.append(df)

    full_val_df = pd.concat(val_dfs, ignore_index=True)
    full_val_df.sort_values(["drive_id", "phone_name", "utcTimeMillis"], inplace=True)
    full_val_df.reset_index(drop=True, inplace=True)

    train_dataset = WindowedDataset(full_train_df, Config.WINDOW_SIZE, mode="train")
    val_dataset = WindowedDataset(full_val_df, Config.WINDOW_SIZE, mode="val")

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    model = SensorFusionTCN(
        num_inputs=Config.NUM_FEATURES,
        num_channels=[Config.HIDDEN_CHANNELS] * Config.NUM_LAYERS,
        kernel_size=Config.KERNEL_SIZE,
        dropout=Config.DROPOUT,
    ).to(Config.DEVICE)

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {Config.DEVICE}...")
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(Config.DEVICE), targets.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(Config.DEVICE), targets.to(
                    Config.DEVICE
                )
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * features.size(0)

        val_loss /= len(val_dataset)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train MAE: {train_loss:.9f} | Val MAE: {val_loss:.9f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(Config.WORKING_DIR, "model_weights.pth"),
            )
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    return model


def generate_submission(test_meta_path, model=None):
    print("Generating Submission...")
    test_meta = pd.read_csv(test_meta_path)

    if model is None:
        model = SensorFusionTCN(
            num_inputs=Config.NUM_FEATURES,
            num_channels=[Config.HIDDEN_CHANNELS] * Config.NUM_LAYERS,
            kernel_size=Config.KERNEL_SIZE,
            dropout=Config.DROPOUT,
        ).to(Config.DEVICE)
        model.load_state_dict(
            torch.load(os.path.join(Config.WORKING_DIR, "model_weights.pth"))
        )

    model.eval()

    unique_drives = test_meta[["drive_id", "phone_name"]].drop_duplicates()
    dfs = []
    for _, row in unique_drives.iterrows():
        df = process_drive_data(
            row["drive_id"],
            row["phone_name"],
            test_meta[
                (test_meta["drive_id"] == row["drive_id"])
                & (test_meta["phone_name"] == row["phone_name"])
            ],
            load_cached_data=True,
        )
        if not df.empty:
            dfs.append(df)

    full_test_df = pd.concat(dfs, ignore_index=True)
    full_test_df.sort_values(["drive_id", "phone_name", "utcTimeMillis"], inplace=True)
    full_test_df.reset_index(drop=True, inplace=True)

    test_dataset = WindowedDataset(full_test_df, Config.WINDOW_SIZE, mode="test")
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    predictions = []

    with torch.no_grad():
        for features, wls_baseline in test_loader:
            features = features.to(Config.DEVICE)
            residuals = model(features).cpu().numpy()
            preds = wls_baseline.numpy() + residuals
            predictions.append(preds)

    all_preds = np.concatenate(predictions, axis=0)

    target_indices = test_dataset.indices
    submission_df = full_test_df.iloc[target_indices].copy()
    submission_df["LatitudeDegrees"] = all_preds[:, 0]
    submission_df["LongitudeDegrees"] = all_preds[:, 1]

    final_sub = submission_df[
        ["tripId", "utcTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ]
    final_sub.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    final_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
