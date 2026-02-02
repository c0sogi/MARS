import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef


# ==========================================
# Configuration
# ==========================================
class Config:
    SEED = 42
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_11"
    SUBMISSION_DIR = "./submission"

    # Data Processing
    # Window Radius 5 means [-5, -4, ..., 0, ..., 4, 5] (11 steps total, approx 1.1s)
    WINDOW_RADIUS = 5
    LAG_STEPS = list(range(-WINDOW_RADIUS, WINDOW_RADIUS + 1))

    # Model Hyperparameters
    INPUT_DIM = 0  # Set dynamically based on feature count
    HIDDEN_DIM = 512
    NUM_BLOCKS = 4
    DROPOUT = 0.2

    # Training Hyperparameters
    BATCH_SIZE = 2048
    LEARNING_RATE = 1e-3
    EPOCHS = 15
    PATIENCE = 3

    # Focal Loss
    FOCAL_ALPHA = 0.75
    FOCAL_GAMMA = 2.0
    POS_WEIGHT = 72.5

    # Debug Mode (set True to run on small data subset)
    DEBUG = False


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ==========================================
# Model Architecture: EC-GRN
# ==========================================
class GatedResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        # Signal path
        h = self.act(self.dense(x))
        h = self.dropout(h)
        # Gating path
        g = torch.sigmoid(self.gate(x))
        # Gated Residual connection
        return x + (h * g)


class ECGRN(nn.Module):
    """
    Entity-Centric Gated Residual Network.
    Accepts wide flattened inputs and uses gated residuals to manage signal flow.
    """

    def __init__(self, input_dim, hidden_dim, num_blocks, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return torch.sigmoid(self.head(x))


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# ==========================================
# Data Processing Pipeline
# ==========================================
def process_tracking_data(tracking_df):
    """
    Generates temporal lags for tracking data entities.
    Returns a dataframe with wide-format features for each step.
    """
    # Sort to ensure correct shifting
    tracking_df = tracking_df.sort_values(
        ["game_play", "nfl_player_id", "step"]
    ).reset_index(drop=True)

    feature_cols = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
    ]
    grp = tracking_df.groupby(["game_play", "nfl_player_id"])

    lagged_dfs = []
    for lag in Config.LAG_STEPS:
        if lag == 0:
            df_lag = tracking_df[feature_cols].copy()
        else:
            # shift(lag) moves data from t-lag to t.
            # If lag is positive (e.g., 5), shift(5) brings previous data forward.
            # If lag is negative (e.g., -5), shift(-5) brings future data backward.
            # We want features at time t to include t+lag.
            # So we use shift(-lag).
            df_lag = grp[feature_cols].shift(-lag)

        df_lag.columns = [f"{c}_lag{lag}" for c in feature_cols]
        lagged_dfs.append(df_lag)

    combined = pd.concat(lagged_dfs, axis=1)

    # Restore keys
    combined["game_play"] = tracking_df["game_play"]
    combined["nfl_player_id"] = tracking_df["nfl_player_id"]
    combined["step"] = tracking_df["step"]

    return combined


def get_data(mode="train", load_cached_data=True):
    """
    Loads and processes data with caching mechanism.
    mode: 'train' or 'test'
    """
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORK_DIR, f"{mode}_features.parquet")

    # 1. Attempt Cache Load
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing {mode} data from scratch...")

    # 2. Load Metadata & Raw Tracking
    if mode == "train":
        meta_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
        val_meta_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "validation.csv"))
        meta_df["is_val"] = 0
        val_meta_df["is_val"] = 1
        labels = pd.concat([meta_df, val_meta_df], ignore_index=True)

        if Config.DEBUG:
            labels = labels.sample(10000, random_state=Config.SEED).reset_index(
                drop=True
            )

        tracking_file = "train_player_tracking.csv"
    else:
        labels = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))
        tracking_file = "test_player_tracking.csv"

    tracking = pd.read_csv(os.path.join(Config.INPUT_DIR, tracking_file))

    # Filter tracking to relevant plays
    relevant_gps = labels["game_play"].unique()
    tracking = tracking[tracking["game_play"].isin(relevant_gps)].copy()

    # 3. Entity-Level Windowing
    print("Generating temporal windows for tracking data...")
    tracking_features = process_tracking_data(tracking)

    # 4. Merge Tracking to Labels
    print("Merging tracking data to labels...")
    labels["nfl_player_id_1"] = pd.to_numeric(
        labels["nfl_player_id_1"], errors="coerce"
    )
    labels["nfl_player_id_2_num"] = pd.to_numeric(
        labels["nfl_player_id_2"], errors="coerce"
    )

    # Merge Player 1
    merged = labels.merge(
        tracking_features,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(columns=["nfl_player_id"])

    # Rename P1 columns
    p1_cols = [
        c
        for c in tracking_features.columns
        if c not in ["game_play", "nfl_player_id", "step"]
    ]
    merged = merged.rename(columns={c: f"{c}_1" for c in p1_cols})

    # Merge Player 2
    merged = merged.merge(
        tracking_features,
        left_on=["game_play", "nfl_player_id_2_num", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(columns=["nfl_player_id"])

    # Rename P2 columns
    merged = merged.rename(columns={c: f"{c}_2" for c in p1_cols})

    # 5. Hybrid Ground Imputation & Relative Features
    print("Computing relative features and imputing ground...")
    is_ground = merged["nfl_player_id_2"] == "G"

    for lag in Config.LAG_STEPS:
        x1 = f"x_position_lag{lag}_1"
        y1 = f"y_position_lag{lag}_1"
        s1 = f"speed_lag{lag}_1"
        d1 = f"direction_lag{lag}_1"

        x2 = f"x_position_lag{lag}_2"
        y2 = f"y_position_lag{lag}_2"
        s2 = f"speed_lag{lag}_2"
        a2 = f"acceleration_lag{lag}_2"
        d2 = f"direction_lag{lag}_2"

        # Impute Ground Physics
        # Ground position = Player position (Distance -> 0)
        merged.loc[is_ground, x2] = merged.loc[is_ground, x1]
        merged.loc[is_ground, y2] = merged.loc[is_ground, y1]
        # Ground velocity/accel = 0 (Relative Speed -> Player Speed)
        merged.loc[is_ground, s2] = 0.0
        merged.loc[is_ground, a2] = 0.0
        merged.loc[is_ground, d2] = 0.0

        # Compute Relative Features
        dx = merged[x1] - merged[x2]
        dy = merged[y1] - merged[y2]
        dist = np.sqrt(dx**2 + dy**2)

        # Calculate Velocity Components (assuming standard angle conventions)
        # Convert direction to radians
        rad1 = np.radians(merged[d1])
        rad2 = np.radians(merged[d2])

        vx1 = merged[s1] * np.sin(rad1)
        vy1 = merged[s1] * np.cos(rad1)
        vx2 = merged[s2] * np.sin(rad2)
        vy2 = merged[s2] * np.cos(rad2)

        dvx = vx1 - vx2
        dvy = vy1 - vy2

        # Closing Speed: Rate of distance decrease
        # Cite Lesson 00007: Numerical Stability (clamped denominator)
        # Cite Lesson 00005: Closing Speed feature
        dot_prod = dx * dvx + dy * dvy
        merged[f"closing_speed_lag{lag}"] = dot_prod / np.maximum(dist, 1e-6)

        merged[f"log_dist_lag{lag}"] = np.log1p(dist)
        merged[f"rel_speed_lag{lag}"] = merged[s1] - merged[s2]

    # Clean up
    drop_cols = [
        "path_endzone",
        "path_sideline",
        "path_all29",
        "datetime",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "nfl_player_id_2_num",
    ]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])
    merged = merged.fillna(0)

    # Save Cache
    print(f"Saving {mode} data to cache...")
    merged.to_parquet(cache_path)

    return merged


class ContactDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ==========================================
# Training & Inference Logic
# ==========================================
def train_model():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    df_all = get_data(mode="train", load_cached_data=True)

    exclude = ["contact_id", "game_play", "step", "contact", "is_val"]
    feature_cols = [c for c in df_all.columns if c not in exclude]
    Config.INPUT_DIM = len(feature_cols)
    print(f"Input Dimension: {Config.INPUT_DIM}")

    # 2. Split & Scale
    train_df = df_all[df_all["is_val"] == 0]
    val_df = df_all[df_all["is_val"] == 1]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    y_train = train_df["contact"].values
    X_val = scaler.transform(val_df[feature_cols].values)
    y_val = val_df["contact"].values

    train_loader = DataLoader(
        ContactDataset(X_train, y_train),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        ContactDataset(X_val, y_val),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    # 3. Model Setup
    model = ECGRN(
        Config.INPUT_DIM, Config.HIDDEN_DIM, Config.NUM_BLOCKS, Config.DROPOUT
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # 4. Training Loop
    best_mcc = -1.0
    best_state = None
    patience = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_b, _ in val_loader:
                val_preds.append(model(X_b.to(device)).cpu().numpy())
        val_preds = np.concatenate(val_preds)

        # Quick MCC check at 0.5
        mcc = matthews_corrcoef(y_val, (val_preds > 0.5).astype(int))
        print(
            f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.6f} | Val MCC (0.5): {mcc:.16f}"
        )

        if mcc > best_mcc:
            best_mcc = mcc
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # 5. Threshold Optimization
    model.load_state_dict(best_state)
    model.eval()

    best_thresh = 0.5
    best_mcc_opt = -1.0

    # Re-predict on val
    val_preds = []
    with torch.no_grad():
        for X_b, _ in val_loader:
            val_preds.append(model(X_b.to(device)).cpu().numpy())
    val_preds = np.concatenate(val_preds)

    for t in np.linspace(0.1, 0.9, 81):
        score = matthews_corrcoef(y_val, (val_preds > t).astype(int))
        if score > best_mcc_opt:
            best_mcc_opt = score
            best_thresh = t

    print(f"Best Threshold: {best_thresh} | Best Val MCC: {best_mcc_opt:.16f}")

    return model, scaler, best_thresh, feature_cols


def generate_submission(model, scaler, threshold, feature_cols):
    print("Generating submission...")
    df_test = get_data(mode="test", load_cached_data=True)

    # Align features
    for c in feature_cols:
        if c not in df_test.columns:
            df_test[c] = 0

    X_test = scaler.transform(df_test[feature_cols].values)
    loader = DataLoader(
        ContactDataset(X_test),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    device = next(model.parameters()).device
    model.eval()

    preds = []
    with torch.no_grad():
        for X_b in loader:
            preds.append(model(X_b.to(device)).cpu().numpy())
    preds = np.concatenate(preds)

    sub_df = pd.DataFrame(
        {
            "contact_id": df_test["contact_id"],
            "contact": (preds > threshold).astype(int).flatten(),
        }
    )

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
