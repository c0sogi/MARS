import os
import random
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler


class Config:
    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_18")
    SUBMISSION_DIR = "./submission"

    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Hyperparameters ---
    SEED = 42
    EPOCHS = 200
    BATCH_SIZE = 256
    LR = 1e-3
    WEIGHT_DECAY = 1e-2

    # Model Architecture
    HIDDEN_SIZE = 256
    GLU_DIM = 256
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # Data
    NUM_WORKERS = 4
    SEQ_LEN = 80  # Breaths are approx 80 steps


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==========================================
# Data Processing & Caching
# ==========================================


def add_features(df):
    """
    Computes physics-inspired features and dynamics.
    """
    # Physics / Interaction Terms
    # Correct integration: u_in * dt. Assuming uniform dt for simplicity or just cumsum u_in
    # The dataset description says 'time_step' is the timestamp.
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0.033)  # approx delta
    df["volume"] = df.groupby("breath_id")["u_in"].cumsum() * df["dt"]

    df["R_u_in"] = df["R"] * df["u_in"]
    df["vol_C"] = df["volume"] / df["C"]
    df["R_div_C"] = df["R"] / df["C"]

    # Dynamics (Lags and Diffs)
    # We use u_in lags. We do NOT use future lags.
    for lag in [1, 2, 3, 4]:
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)

    # Finite differences
    df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff().fillna(0)
    df["u_in_diff2"] = df.groupby("breath_id")["u_in_diff1"].diff().fillna(0)

    # One-hot encoding R and C could be useful, but we treat them as continuous per "Signal Fidelity" idea
    # However, R and C are constant per breath.

    return df


def process_data(load_cached_data=True):
    """
    Loads data, generates features, scales continuous variables, and caches the result.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    train_cache = os.path.join(Config.CACHE_DIR, "train_processed.npz")
    val_cache = os.path.join(Config.CACHE_DIR, "val_processed.npz")
    test_cache = os.path.join(Config.CACHE_DIR, "test_processed.npz")
    scaler_cache = os.path.join(Config.CACHE_DIR, "scaler_params.npz")

    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print("Loading cached data...")
        train_data = np.load(train_cache)
        val_data = np.load(val_cache)
        test_data = np.load(test_cache)
        return (
            (train_data["X"], train_data["y"], train_data["u_out"]),
            (val_data["X"], val_data["y"], val_data["u_out"]),
            (test_data["X"], test_data["ids"]),
        )

    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META)
    val_meta = pd.read_csv(Config.VAL_META)
    test_meta = pd.read_csv(Config.TEST_META)

    # Load Raw Data
    # To save memory, we can load full train and split based on breath_id from metadata
    df_train_full = pd.read_csv(Config.TRAIN_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Split Train/Val based on metadata breath_ids
    train_breath_ids = train_meta["breath_id"].unique()
    val_breath_ids = val_meta["breath_id"].unique()

    df_train = df_train_full[df_train_full["breath_id"].isin(train_breath_ids)].copy()
    df_val = df_train_full[df_train_full["breath_id"].isin(val_breath_ids)].copy()

    del df_train_full
    gc.collect()

    # Feature Engineering
    print("Generating features...")
    df_train = add_features(df_train)
    df_val = add_features(df_val)
    df_test = add_features(df_test)

    # Define Feature Columns
    # Exclude IDs, targets, and u_out (handled separately)
    exclude = ["id", "breath_id", "pressure", "u_out", "dt"]
    feature_cols = [c for c in df_train.columns if c not in exclude]

    print(f"Continuous Features: {feature_cols}")

    # Scaling - RobustScaler on Continuous features only
    print("Scaling features...")
    scaler = RobustScaler()

    # Fit on Train
    X_train_cont = scaler.fit_transform(df_train[feature_cols].values)
    X_val_cont = scaler.transform(df_val[feature_cols].values)
    X_test_cont = scaler.transform(df_test[feature_cols].values)

    # Prepare Binary u_out (Raw)
    u_out_train = df_train["u_out"].values.reshape(-1, 1)
    u_out_val = df_val["u_out"].values.reshape(-1, 1)
    u_out_test = df_test["u_out"].values.reshape(-1, 1)

    # Concatenate: [Continuous, Binary]
    X_train = np.hstack([X_train_cont, u_out_train])
    X_val = np.hstack([X_val_cont, u_out_val])
    X_test = np.hstack([X_test_cont, u_out_test])

    # Targets
    y_train = df_train["pressure"].values
    y_val = df_val["pressure"].values

    # IDs for test
    test_ids = df_test["id"].values

    # Reshape for LSTM: (N_breaths, 80, N_features)
    # We assume data is sorted by breath_id and time_step (standard in this dataset)
    # Each breath is exactly 80 steps.

    def reshape_to_seq(X, y=None):
        num_breaths = X.shape[0] // 80
        X_seq = X.reshape(num_breaths, 80, -1)
        if y is not None:
            y_seq = y.reshape(num_breaths, 80)
            return X_seq, y_seq
        return X_seq

    print("Reshaping tensors...")
    X_train_seq, y_train_seq = reshape_to_seq(X_train, y_train)
    X_val_seq, y_val_seq = reshape_to_seq(X_val, y_val)
    X_test_seq = reshape_to_seq(X_test)

    # Extract u_out sequences for loss weighting
    u_out_train_seq = X_train_seq[:, :, -1]  # Last column is u_out
    u_out_val_seq = X_val_seq[:, :, -1]

    # Save to Cache
    print("Saving to cache...")
    np.savez(train_cache, X=X_train_seq, y=y_train_seq, u_out=u_out_train_seq)
    np.savez(val_cache, X=X_val_seq, y=y_val_seq, u_out=u_out_val_seq)
    np.savez(test_cache, X=X_test_seq, ids=test_ids)

    # Save scaler params (mean/scale) just in case
    np.savez(scaler_cache, center=scaler.center_, scale=scaler.scale_)

    return (
        (X_train_seq, y_train_seq, u_out_train_seq),
        (X_val_seq, y_val_seq, u_out_val_seq),
        (X_test_seq, test_ids),
    )


class VentilatorDataset(Dataset):
    def __init__(self, X, y=None, u_out=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.u_out = (
            torch.tensor(u_out, dtype=torch.float32) if u_out is not None else None
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx], self.u_out[idx]
        return self.X[idx]


# ==========================================
# Model: WCMI-BiLSTM
# ==========================================


class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        out = self.fc(x)
        out, gate = out.chunk(2, dim=-1)
        return out * torch.sigmoid(gate)


class WCMI_BiLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.hidden_size = Config.HIDDEN_SIZE

        # Wide Monolithic Context Extractor
        self.glu = GLU(input_dim, Config.GLU_DIM)

        # Injection Payload Dimension: GLU output + Raw Input (Identity Path)
        self.injection_dim = Config.GLU_DIM + input_dim

        # Deep Recurrent Backbone
        self.layers = nn.ModuleList()
        for _ in range(Config.NUM_LAYERS):
            self.layers.append(
                nn.LSTM(
                    input_size=(
                        self.injection_dim
                        if _ == 0
                        else (self.hidden_size * 2 + self.injection_dim)
                    ),
                    hidden_size=self.hidden_size,
                    batch_first=True,
                    bidirectional=True,
                )
            )

        self.lns = nn.ModuleList(
            [nn.LayerNorm(self.hidden_size * 2) for _ in range(Config.NUM_LAYERS)]
        )
        self.dropout = nn.Dropout(Config.DROPOUT)

        self.head = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, x):
        # x: [Batch, Seq, Feat]

        # 1. Context Extraction
        context = self.glu(x)

        # 2. Construct Injection Payload (Context + Identity)
        injection = torch.cat([context, x], dim=-1)

        curr_input = injection

        # 3. Deep Backbone with Deep Injection
        for i, lstm in enumerate(self.layers):
            if i > 0:
                # Concatenate injection payload to previous layer output
                curr_input = torch.cat([curr_input, injection], dim=-1)

            output, _ = lstm(curr_input)
            output = self.lns[i](output)
            output = self.dropout(output)
            curr_input = output

        # 4. Head
        pred = self.head(curr_input).squeeze(-1)
        return pred


# ==========================================
# Training & Evaluation
# ==========================================


def weighted_l1_loss(pred, target, u_out):
    """
    Weighted L1 Loss: 1.0 for Inspiratory (u_out=0), 0.1 for Expiratory (u_out=1).
    """
    error = torch.abs(pred - target)
    weights = 1.0 * (1 - u_out) + 0.1 * u_out
    return (error * weights).mean()


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for X, y, u_out in loader:
        X, y, u_out = X.to(device), y.to(device), u_out.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = weighted_l1_loss(pred, y, u_out)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_mae = 0
    count = 0

    # Metric: MAE on Inspiratory Phase ONLY (u_out == 0)
    with torch.no_grad():
        for X, y, u_out in loader:
            X, y, u_out = X.to(device), y.to(device), u_out.to(device)
            pred = model(X)

            # Mask for inspiratory phase
            mask = u_out == 0
            if mask.sum() > 0:
                mae = torch.abs(pred[mask] - y[mask]).sum()
                total_mae += mae.item()
                count += mask.sum().item()

    return total_mae / count if count > 0 else 0


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for X in loader:
            X = X.to(device)
            pred = model(X)
            preds.append(pred.cpu().numpy().flatten())
    return np.concatenate(preds)


def run_job():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Prepare Data
    (X_train, y_train, u_out_train), (X_val, y_val, u_out_val), (X_test, test_ids) = (
        process_data(load_cached_data=True)
    )

    train_dataset = VentilatorDataset(X_train, y_train, u_out_train)
    val_dataset = VentilatorDataset(X_val, y_val, u_out_val)
    test_dataset = VentilatorDataset(X_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Model
    input_dim = X_train.shape[2]
    model = WCMI_BiLSTM(input_dim).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing over total steps
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    # 3. Training Loop
    best_mae = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_mae = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MAE (Inspiratory): {val_mae:.6f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                model.state_dict(), os.path.join(Config.WORKING_DIR, "best_model.pth")
            )

    print(f"Best Val MAE: {best_mae:.6f}")

    # 4. Inference
    print("Generating submission...")
    model.load_state_dict(
        torch.load(os.path.join(Config.WORKING_DIR, "best_model.pth"))
    )
    predictions = predict(model, test_loader, device)

    # Create submission dataframe
    # Note: predictions are flattened (N_breaths * 80), matching the test_ids order (assuming standard sorting)
    # The test_ids from process_data are just the 'id' column from test.csv

    # The test set in process_data was reshaped to (N, 80). Flattening it puts it back in time order.
    # Ensure lengths match
    if len(predictions) != len(test_ids):
        print(
            f"Warning: Prediction length {len(predictions)} != ID length {len(test_ids)}"
        )

    submission = pd.DataFrame({"id": test_ids, "pressure": predictions})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
