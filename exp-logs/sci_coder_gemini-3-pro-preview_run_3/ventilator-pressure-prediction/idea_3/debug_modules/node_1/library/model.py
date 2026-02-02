import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from sklearn.preprocessing import StandardScaler
from library.config import Config

# -----------------------------------------------------------------------------
# 1. Model Architecture: Parallel TCN-LSTM Hybrid
# -----------------------------------------------------------------------------


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
        # First dilated convolution
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

        # Second dilated convolution
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

        # Residual connection
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


class TCNBranch(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCNBranch, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # Padding such that output length equals input length (causal-like structure)
            padding = (kernel_size - 1) * dilation_size

            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, Input_Channels, Seq_Len)
        return self.network(x)


class LSTMBranch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout):
        super(LSTMBranch, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Features)
        output, _ = self.lstm(x)
        return output


class ParallelTCNLSTM(nn.Module):
    def __init__(self, config):
        super(ParallelTCNLSTM, self).__init__()
        self.input_dim = config["input_dim"]

        # --- TCN Branch (Fast Dynamics) ---
        tcn_channels = [config["tcn_channels"]] * config["tcn_levels"]
        self.tcn = TCNBranch(
            num_inputs=self.input_dim,
            num_channels=tcn_channels,
            kernel_size=config["tcn_kernel_size"],
            dropout=config["tcn_dropout"],
        )
        tcn_out_dim = tcn_channels[-1]

        # --- LSTM Branch (Slow Integration) ---
        self.lstm = LSTMBranch(
            input_size=self.input_dim,
            hidden_size=config["lstm_hidden_dim"],
            num_layers=config["lstm_layers"],
            bidirectional=config["lstm_bidirectional"],
            dropout=config["lstm_dropout"],
        )
        lstm_out_dim = config["lstm_hidden_dim"] * (
            2 if config["lstm_bidirectional"] else 1
        )

        # --- Fusion Head (Direct Physics Injection) ---
        # Concatenate: TCN_Out + LSTM_Out + Original_Input
        fusion_dim = tcn_out_dim + lstm_out_dim + self.input_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, config["fc_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["fc_hidden_dim"], 1),
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)

        # 1. TCN Path (Requires Channel-First)
        x_tcn_in = x.permute(0, 2, 1)
        tcn_out = self.tcn(x_tcn_in)
        tcn_out = tcn_out.permute(0, 2, 1)  # Back to (Batch, Seq_Len, Channels)

        # 2. LSTM Path
        lstm_out = self.lstm(x)

        # 3. Fusion with Direct Physics Injection
        # Concatenate latent representations with the raw physical features
        combined = torch.cat([tcn_out, lstm_out, x], dim=2)

        # 4. Prediction
        pressure = self.head(combined)
        return pressure


# -----------------------------------------------------------------------------
# 2. Data Pipeline & Feature Engineering
# -----------------------------------------------------------------------------


class VentilatorDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def compute_features(df):
    """
    Computes PID state and physics features.
    Assumes df is sorted by breath_id and time_step.
    Reshapes to (N_breaths, 80, Cols) for vectorized efficiency.
    """
    # Ensure correct sorting
    # df = df.sort_values(['breath_id', 'time_step']) # Assumed sorted by metadata generator

    # Reshape to (N_breaths, 80, N_cols)
    # 80 is the fixed breath length in this dataset
    n_breaths = len(df) // 80

    # Extract columns as numpy arrays
    u_in = df["u_in"].values.reshape(n_breaths, 80)
    u_out = df["u_out"].values.reshape(n_breaths, 80)
    R = df["R"].values.reshape(n_breaths, 80)
    C = df["C"].values.reshape(n_breaths, 80)
    time_step = df["time_step"].values.reshape(n_breaths, 80)

    # --- Feature Engineering ---

    # Integral (Volume Proxy)
    # dt is roughly constant, but u_in * dt is volume. Here we just sum u_in.
    u_in_cumsum = np.cumsum(u_in, axis=1)

    # Derivative (Flow Acceleration)
    # Use gradient or diff. Diff is simpler. Pad with 0 at start.
    u_in_diff1 = np.diff(u_in, axis=1, prepend=0)
    u_in_diff2 = np.diff(u_in_diff1, axis=1, prepend=0)

    # Interactions
    R_u_in = R * u_in
    vol_C_ratio = u_in_cumsum / C

    # Construct Feature Matrix
    # Order must match Config.FEATURE_COLS
    # ["time_step", "u_in", "u_out", "R", "C", "u_in_cumsum", "u_in_diff1", "u_in_diff2", "R_u_in", "vol_C_ratio"]

    features = np.stack(
        [
            time_step,
            u_in,
            u_out,
            R,
            C,
            u_in_cumsum,
            u_in_diff1,
            u_in_diff2,
            R_u_in,
            vol_C_ratio,
        ],
        axis=2,
    )  # Shape: (N_breaths, 80, 10)

    return features


def prepare_data(load_cached_data=True):
    """
    Orchestrates data loading, feature engineering, caching, and scaling.
    """
    # Paths
    cache_dir = Config.CACHE_DIR
    train_x_path = os.path.join(cache_dir, "train_x.npy")
    train_y_path = os.path.join(cache_dir, "train_y.npy")
    val_x_path = os.path.join(cache_dir, "val_x.npy")
    val_y_path = os.path.join(cache_dir, "val_y.npy")
    test_x_path = os.path.join(cache_dir, "test_x.npy")
    stats_path = os.path.join(cache_dir, "scaler_stats.npy")

    # Check Cache
    if Config.FORCE_RECOMPUTE:
        load_cached_data = False

    if load_cached_data and os.path.exists(train_x_path):
        print("Loading cached data...")
        X_train = np.load(train_x_path)
        y_train = np.load(train_y_path)
        X_val = np.load(val_x_path)
        y_val = np.load(val_y_path)
        X_test = np.load(test_x_path)
        return X_train, y_train, X_val, y_val, X_test

    print("Computing features from scratch...")

    # Load Raw Metadata
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    if Config.DEBUG:
        print(f"DEBUG MODE: Using only {Config.DEBUG_BREATHS} breaths")
        train_df = train_df.iloc[: Config.DEBUG_BREATHS * 80]
        val_df = val_df.iloc[: Config.DEBUG_BREATHS * 80]
        test_df = test_df.iloc[: Config.DEBUG_BREATHS * 80]

    # Compute Features (Returns [N, 80, F])
    print("Engineering features for Train...")
    X_train = compute_features(train_df)
    y_train = train_df["pressure"].values.reshape(-1, 80, 1)

    print("Engineering features for Val...")
    X_val = compute_features(val_df)
    y_val = val_df["pressure"].values.reshape(-1, 80, 1)

    print("Engineering features for Test...")
    X_test = compute_features(test_df)

    # Scaling
    # We flatten, fit scaler, reshape back.
    # Exclude u_out (binary) from scaling? Ideally yes, but standard scaler handles 0/1 fine (just centers it).
    print("Fitting Scaler...")
    N_train, L, F = X_train.shape
    scaler = StandardScaler()

    # Fit on train
    X_train_flat = X_train.reshape(-1, F)
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_train = X_train_flat.reshape(N_train, L, F)

    # Transform Val
    N_val, _, _ = X_val.shape
    X_val = scaler.transform(X_val.reshape(-1, F)).reshape(N_val, L, F)

    # Transform Test
    N_test, _, _ = X_test.shape
    X_test = scaler.transform(X_test.reshape(-1, F)).reshape(N_test, L, F)

    # Save Cache
    print("Saving cache...")
    np.save(train_x_path, X_train)
    np.save(train_y_path, y_train)
    np.save(val_x_path, X_val)
    np.save(val_y_path, y_val)
    np.save(test_x_path, X_test)

    # Save scaler stats for reference (mean, scale)
    np.save(stats_path, np.stack([scaler.mean_, scaler.scale_]))

    return X_train, y_train, X_val, y_val, X_test


# -----------------------------------------------------------------------------
# 3. Training & Evaluation
# -----------------------------------------------------------------------------


def masked_mae_loss(y_pred, y_true, u_out):
    """
    Computes MAE only for the inspiratory phase (u_out == 0).
    Assumes u_out is standardized, so u_out == 0 corresponds to u_out < 0.
    """
    # u_out is (Batch, Seq, 1) or (Batch, Seq)
    if u_out.dim() == 2:
        u_out = u_out.unsqueeze(-1)

    # Since u_out is standardized and mean is ~0.6, u_out=0 becomes negative.
    mask = (u_out < 0).float()
    loss = torch.abs(y_pred - y_true) * mask
    return loss.sum() / (mask.sum() + 1e-8)


def train_model(config=Config.HYPERPARAMS):
    # 1. Prepare Data
    X_train, y_train, X_val, y_val, X_test = prepare_data(
        load_cached_data=not Config.FORCE_RECOMPUTE
    )

    train_dataset = VentilatorDataset(X_train, y_train)
    val_dataset = VentilatorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    # 2. Setup Model
    device = torch.device(config["device"])
    model = ParallelTCNLSTM(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
        verbose=True,
    )

    # 3. Training Loop
    best_val_mae = float("inf")
    early_stop_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        train_steps = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Extract u_out for masking (Index 2 in FEATURE_COLS)
            # We standardized u_out, so it's not exactly 0/1 anymore.
            # However, since it's binary, standardized values will cluster around two points.
            # We can recover the binary mask by checking > 0 or similar, but safer to rely on the fact
            # that u_out=0 is likely the smaller value.
            # Actually, let's look at the scaler logic.
            # Ideally we shouldn't scale u_out for mask usage, but we did.
            # Workaround: The raw u_out is 0 or 1. After scaling: (x - mean)/std.
            # If u_out was 0, scaled value is (0 - mean)/std < 0.
            # If u_out was 1, scaled value is (1 - mean)/std > 0.
            # So u_out_raw == 0 corresponds to u_out_scaled < threshold (roughly 0).
            # Let's use a threshold check.

            u_out_scaled = X_batch[:, :, 2:3]
            # Thresholding: Since u_out is 0 or 1, and mean is ~0.6, 0 is mapped to negative, 1 to positive.
            # We want mask where u_out == 0. So u_out_scaled < 0.
            # Let's verify: mean=0.62. 0 -> -1.2, 1 -> 0.7. Cutoff 0 is safe.
            mask_condition = u_out_scaled < 0  # True for inspiratory

            # Forward
            preds = model(X_batch)

            # Loss: Calculate MAE manually with mask
            loss = torch.abs(preds - y_batch) * mask_condition.float()
            loss = loss.sum() / (mask_condition.float().sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        avg_train_loss = train_loss / train_steps

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                u_out_scaled = X_batch[:, :, 2:3]
                mask_condition = u_out_scaled < 0

                preds = model(X_batch)
                loss = torch.abs(preds - y_batch) * mask_condition.float()
                loss = loss.sum() / (mask_condition.float().sum() + 1e-8)

                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{config['epochs']} | Train MAE: {avg_train_loss:.6f} | Val MAE: {avg_val_loss:.6f}"
        )

        # Checkpointing
        if avg_val_loss < best_val_mae:
            best_val_mae = avg_val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= config["patience"]:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val MAE: {best_val_mae:.6f}")

    # 4. Generate Submission
    generate_submission(model, X_test, device)


def generate_submission(model, X_test, device):
    print("Generating submission...")
    # Load best weights
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.eval()

    test_dataset = VentilatorDataset(X_test)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.HYPERPARAMS["batch_size"], shuffle=False
    )

    predictions = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            predictions.append(preds.cpu().numpy().flatten())

    all_preds = np.concatenate(predictions)

    # Load sample submission to get IDs
    sub = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))
    # The test.csv in metadata is the raw test file.
    # We need to ensure the order matches.
    # Our X_test was generated from Config.TEST_PATH which is metadata/test.csv.
    # So the order is preserved.

    sub["pressure"] = all_preds

    # Format: id, pressure
    output = sub[["id", "pressure"]]
    output.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
