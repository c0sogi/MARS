import os
import random
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

# ==========================================
# Configuration
# ==========================================


class Config:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Model Hyperparameters
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 4
    LSTM_DROPOUT = 0.1

    # Training Hyperparameters
    SEED = 42
    EPOCHS = 50
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05
    MAX_LR = 1e-3

    # Loss Weights
    W_INSPIRATORY = 1.0
    W_EXPIRATORY = 0.1

    # Feature Engineering
    USE_LAG_FEATURES = True
    USE_DIFF_FEATURES = True

    def __init__(self):
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)


# ==========================================
# Utilities
# ==========================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==========================================
# Data Processing
# ==========================================


def add_features(df):
    """
    Adds physics-informed features and time-series derivatives.
    """
    # 1. Physics-Based Integration (Air Volume)
    # Group by breath_id to calculate cumulative sum of u_in correctly
    # Note: We assume df is sorted by breath_id and time_step
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

    # 2. Equation of Motion Interactions
    df["R_u_in"] = df["R"] * df["u_in"]
    df["vol_C"] = df["u_in_cumsum"] / df["C"]

    # 3. Explicit Dynamics (Lags and Diffs)
    # We use shift/diff within groups.
    # Optimization: Since data is sorted, we can use simple shift and mask boundaries,
    # but groupby is safer for correctness.
    grp = df.groupby("breath_id")

    df["u_in_lag1"] = grp["u_in"].shift(1).fillna(0)
    df["u_in_lag2"] = grp["u_in"].shift(2).fillna(0)

    df["u_in_diff1"] = grp["u_in"].diff(1).fillna(0)
    df["u_in_diff2"] = grp["u_in"].diff(2).fillna(0)

    # 4. Time delta
    df["dt"] = grp["time_step"].diff(1).fillna(0)

    return df


def get_processed_data(config, split="train", load_cached_data=True):
    """
    Loads, processes, and caches data.
    Returns: X (features), y (targets), u_out (auxiliary for loss)
    """
    cache_file_X = os.path.join(config.WORKING_DIR, f"X_{split}.npy")
    cache_file_y = os.path.join(config.WORKING_DIR, f"y_{split}.npy")
    cache_file_uout = os.path.join(config.WORKING_DIR, f"u_out_{split}.npy")
    scaler_file = os.path.join(config.WORKING_DIR, "scaler_params.npy")

    if load_cached_data and os.path.exists(cache_file_X):
        print(f"Loading cached {split} data...")
        X = np.load(cache_file_X)
        u_out = np.load(cache_file_uout)
        y = np.load(cache_file_y) if split != "test" else None
        return X, y, u_out

    print(f"Processing {split} data from scratch...")

    # Load Metadata to filter breaths
    if split == "train":
        meta_df = pd.read_csv(config.TRAIN_META)
        source_df = pd.read_csv(config.TRAIN_PATH)
    elif split == "val":
        meta_df = pd.read_csv(config.VAL_META)
        source_df = pd.read_csv(config.TRAIN_PATH)
    else:  # test
        meta_df = pd.read_csv(config.TEST_META)
        source_df = pd.read_csv(config.TEST_PATH)

    # Filter source data based on breath_ids in metadata
    target_breaths = meta_df["breath_id"].unique()
    df = source_df[source_df["breath_id"].isin(target_breaths)].copy()

    # Sort to ensure time order (critical for LSTM)
    df = df.sort_values(["breath_id", "id"]).reset_index(drop=True)

    # Feature Engineering
    df = add_features(df)

    # Define Feature Columns
    feature_cols = [
        "time_step",
        "u_in",
        "u_in_cumsum",
        "R",
        "C",
        "R_u_in",
        "vol_C",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff1",
        "u_in_diff2",
        "dt",
    ]
    # u_out is handled separately for loss weighting

    # Scaling
    if split == "train":
        scaler = RobustScaler()
        data_matrix = df[feature_cols].values
        data_matrix = scaler.fit_transform(data_matrix)
        # Save scaler params (center and scale)
        np.save(scaler_file, {"center": scaler.center_, "scale": scaler.scale_})
    else:
        # Load scaler params manually to avoid pickling the whole object
        if not os.path.exists(scaler_file):
            raise FileNotFoundError("Scaler params not found. Process train set first.")
        params = np.load(scaler_file, allow_pickle=True).item()
        data_matrix = df[feature_cols].values
        data_matrix = (data_matrix - params["center"]) / params["scale"]

    # Add u_out as a feature?
    # The prompt implies u_out is a control input. We append it to features.
    # It's binary, so scaling isn't strictly necessary but standardizing is fine.
    # We'll just append raw u_out to the scaled features.
    u_out_col = df["u_out"].values.reshape(-1, 1)
    X_flat = np.hstack([data_matrix, u_out_col])

    # Reshape to (Num_Breaths, 80, Num_Features)
    # Each breath has exactly 80 time steps
    num_breaths = len(df) // 80
    num_features = X_flat.shape[1]

    X = X_flat.reshape(num_breaths, 80, num_features)
    u_out = df["u_out"].values.reshape(num_breaths, 80)

    if split != "test":
        y = df["pressure"].values.reshape(num_breaths, 80)
    else:
        y = None

    # Cache
    np.save(cache_file_X, X)
    np.save(cache_file_uout, u_out)
    if y is not None:
        np.save(cache_file_y, y)

    return X, y, u_out


class VentilatorDataset(Dataset):
    def __init__(self, X, y=None, u_out=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.u_out = (
            torch.tensor(u_out, dtype=torch.float32) if u_out is not None else None
        )
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = {"X": self.X[idx]}
        if self.u_out is not None:
            data["u_out"] = self.u_out[idx]
        if self.y is not None:
            data["y"] = self.y[idx]
        return data


# ==========================================
# Model Architecture
# ==========================================


class ResidualMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(
            hidden_dim, input_dim
        )  # Project back to input dim for residual?
        # Usually residual keeps dimension. If input_dim != hidden_dim, we need projection.
        # Let's assume we project up first, then residual is on hidden_dim.

        # Revised Residual Block: Input -> Linear(H) -> GELU -> Linear(H) -> Add -> Output
        # We need an initial projection if input_dim != hidden_dim

    def forward(self, x):
        # This is a simplified residual block logic implemented in the main model for flexibility
        pass


class WideAndDeepBiLSTM(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.hidden_size = config.LSTM_HIDDEN_SIZE

        # --- Deep Stream (Recurrent) ---
        # Input Injection: We will have 4 LSTM layers.
        # Layer 1 Input: raw_features
        # Layer 2 Input: cat(layer1_out, raw_features)
        # etc.

        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        current_input_dim = input_dim
        for _ in range(config.LSTM_LAYERS):
            lstm = nn.LSTM(
                input_size=current_input_dim,
                hidden_size=self.hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            self.lstm_layers.append(lstm)
            self.layer_norms.append(nn.LayerNorm(self.hidden_size * 2))
            # Next layer input is output of this layer + raw features
            current_input_dim = (self.hidden_size * 2) + input_dim

        # --- Wide Stream (Instantaneous) ---
        # Residual MLP
        self.mlp_proj = nn.Linear(input_dim, config.MLP_HIDDEN_SIZE)
        self.mlp_blocks = nn.ModuleList()
        for _ in range(config.MLP_LAYERS):
            block = nn.Sequential(
                nn.Linear(config.MLP_HIDDEN_SIZE, config.MLP_HIDDEN_SIZE),
                nn.GELU(),
                nn.Dropout(config.MLP_DROPOUT),
                nn.Linear(config.MLP_HIDDEN_SIZE, config.MLP_HIDDEN_SIZE),
                nn.Dropout(config.MLP_DROPOUT),
            )
            self.mlp_blocks.append(block)

        # --- Fusion Head ---
        # Input: Last LSTM layer output (H*2) + MLP output (H_mlp)
        fusion_dim = (self.hidden_size * 2) + config.MLP_HIDDEN_SIZE
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
        )

    def forward(self, x):
        # x shape: (Batch, Seq, Feat)

        # 1. Deep Recurrent Stream
        h = x
        lstm_out = None

        for i, lstm in enumerate(self.lstm_layers):
            if i > 0:
                # Input Injection: Concatenate raw x with previous hidden state
                h = torch.cat([lstm_out, x], dim=-1)

            lstm_out, _ = lstm(h)
            lstm_out = self.layer_norms[i](lstm_out)

        # 2. Wide Instantaneous Stream
        mlp_out = self.mlp_proj(x)
        for block in self.mlp_blocks:
            residual = mlp_out
            out = block(mlp_out)
            mlp_out = out + residual  # Residual connection

        # 3. Fusion
        combined = torch.cat([lstm_out, mlp_out], dim=-1)
        pred = self.head(combined)

        return pred.squeeze(-1)


# ==========================================
# Training Loop
# ==========================================


def weighted_l1_loss(pred, target, u_out, config):
    # u_out: 0 is inspiratory (weight 1.0), 1 is expiratory (weight 0.1)
    # Weights: 1.0 - (1 - 0.1) * u_out  -> if u_out=0, w=1.0. if u_out=1, w=0.1
    weights = 1.0 - (1.0 - config.W_EXPIRATORY) * u_out
    loss = torch.abs(pred - target) * weights
    return loss.mean()


def train_epoch(model, loader, optimizer, scheduler, device, config):
    model.train()
    total_loss = 0

    for batch in loader:
        X = batch["X"].to(device)
        y = batch["y"].to(device)
        u_out = batch["u_out"].to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = weighted_l1_loss(pred, y, u_out, config)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device, config):
    model.eval()
    total_mae = 0
    count = 0

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            pred = model(X)

            # Metric: MAE only on inspiratory phase (u_out == 0)
            mask = u_out == 0
            mae = torch.abs(pred[mask] - y[mask]).sum()

            total_mae += mae.item()
            count += mask.sum().item()

    return total_mae / count


def run_training(config):
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    X_train, y_train, u_out_train = get_processed_data(config, "train")
    X_val, y_val, u_out_val = get_processed_data(config, "val")

    train_dataset = VentilatorDataset(X_train, y_train, u_out_train)
    val_dataset = VentilatorDataset(X_val, y_val, u_out_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = WideAndDeepBiLSTM(input_dim=X_train.shape[2], config=config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.MAX_LR,
        epochs=config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )

    best_mae = float("inf")
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device} for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, config
        )
        val_mae = validate(model, val_loader, device, config)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} - Train Loss: {train_loss:.6f} - Val MAE: {val_mae:.6f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val MAE: {best_mae}")


# ==========================================
# Inference
# ==========================================


def generate_submission(config):
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    X_test, _, u_out_test = get_processed_data(config, "test")
    test_dataset = VentilatorDataset(X_test, u_out=u_out_test)
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Load Model
    model = WideAndDeepBiLSTM(input_dim=X_test.shape[2], config=config).to(device)
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("Model file not found! Skipping inference.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            X = batch["X"].to(device)
            pred = model(X)
            predictions.append(pred.cpu().numpy().flatten())

    predictions = np.concatenate(predictions)

    # Load Test Metadata to map predictions to IDs
    test_meta = pd.read_csv(config.TEST_META)

    # Note: The model predicts 80 steps per breath. The test set is also organized by breath.
    # We need to ensure alignment.
    # The test_meta is sorted by breath_id, then id. Our predictions follow the same order
    # because get_processed_data sorts by breath_id, id.

    # Ensure lengths match
    if len(predictions) != len(test_meta):
        print(
            f"Warning: Prediction count {len(predictions)} != Metadata count {len(test_meta)}"
        )

    submission = pd.DataFrame({"id": test_meta["id"], "pressure": predictions})

    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


# ==========================================
# Execution Entry Points
# ==========================================


def main_train():
    config = Config()
    run_training(config)


def main_submit():
    config = Config()
    generate_submission(config)
