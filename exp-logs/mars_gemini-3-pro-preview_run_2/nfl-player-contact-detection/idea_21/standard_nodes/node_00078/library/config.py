import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
import joblib


# ==========================================
# Configuration
# ==========================================
class Config:
    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_21"
    OUTPUT_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data Processing
    WINDOW_SIZE = 5  # t-5 to t+5 (Total 11 steps)
    SEED = 42

    # Feature Columns
    # Raw tracking columns to load
    TRACKING_COLS = [
        "game_play",
        "step",
        "nfl_player_id",
        "datetime",
        "x_position",
        "y_position",
        "speed",
        "distance",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]

    # Engineered Kinematic Features (per timestep)
    # P1 and P2 raw + Relative features
    # Note: We will generate these names dynamically in the code, but defining base here
    KINEMATIC_BASE_COLS = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "direction",
        "orientation",
        "distance",
        "sa",
    ]

    # Visual Features (from helmets)
    VISUAL_COLS = ["left", "width", "top", "height"]

    # Model Hyperparameters
    HIDDEN_DIM = 256
    DROPOUT = 0.1
    VISUAL_HIDDEN_DIM = 64

    # Training Hyperparameters
    BATCH_SIZE = 4096
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    PATIENCE = 3

    # Loss
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # Paths
    TRAIN_LABELS = os.path.join(INPUT_DIR, "train_labels.csv")
    TRAIN_TRACKING = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TRAIN_HELMETS = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")

    TEST_TRACKING = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    TEST_HELMETS = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    METADATA_TRAIN = os.path.join(METADATA_DIR, "train.csv")
    METADATA_VAL = os.path.join(METADATA_DIR, "validation.csv")
    METADATA_TEST = os.path.join(METADATA_DIR, "test.csv")

    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.joblib")


# ==========================================
# Data Processing
# ==========================================


def angular_diff(a, b):
    """Computes shortest arc difference between angles."""
    diff = np.abs(a - b)
    return np.minimum(diff, 360 - diff)


def prepare_tracking_data(tracking_df):
    """
    Preprocesses tracking data:
    1. Sorts by game_play, step.
    2. Generates lag features (Windowing).
    """
    # Sort for windowing
    tracking_df = tracking_df.sort_values(
        ["game_play", "nfl_player_id", "step"]
    ).reset_index(drop=True)

    # Features to window
    feature_cols = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "direction",
        "orientation",
        "distance",
        "sa",
    ]

    # Generate lags
    # We want t-5 to t+5.
    # We perform this via shifting within groups.
    # Optimization: Use pivot/unstack or simply shift if data is dense.
    # Given memory constraints and pandas speed, groupby shift is okay but slow.
    # Faster approach: Ensure strict sorting and use shift on the whole array with mask handling.

    # However, to avoid complexity with missing steps, we will assume tracking is mostly dense
    # and handle alignment during the merge step or use a simpler lag generation.
    # Actually, the "Entity-First" lesson suggests generating features on the tracking DF first.

    # Let's create a wide dataframe.
    # Since we need to merge P1 and P2 later, we should keep the format long but with windowed columns.

    # To save memory/time, we will just keep the raw columns and do the windowing lookup
    # or join logic. BUT, doing 11 joins is expensive.
    # Better: Create the lags now.

    df_lagged = tracking_df[["game_play", "nfl_player_id", "step"]].copy()

    # Group object for shifting
    grp = tracking_df.groupby(["game_play", "nfl_player_id"])

    for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        suffix = f"_lag_{lag}"
        for col in feature_cols:
            # lag < 0 means future (shift negative), lag > 0 means past (shift positive)
            # Standard shift: shift(1) moves t to t+1 (gets previous).
            # We want t to contain t-5...t+5.
            # So for lag -5 (past), we want value at t-5. This is shift(5).
            # For lag +5 (future), we want value at t+5. This is shift(-5).
            # Let's align naming: lag_m5 is t-5.

            # Note: shift(k) takes value from index i-k and puts it at i.
            # So shift(5) puts t-5 at t.

            shifted = grp[col].shift(lag)
            # Fill edges with nearest valid observation (ffill/bfill) or 0
            # We'll fill with 0 for simplicity and robustness
            shifted = shifted.fillna(0)
            df_lagged[f"{col}{suffix}"] = shifted

    return df_lagged


def process_dataset(mode="train", load_cached_data=True):
    """
    Main data processing function with caching.
    mode: 'train' (includes val), 'test'
    """
    cache_file = os.path.join(Config.WORKING_DIR, f"{mode}_processed.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"Processing {mode} data from scratch...")

    # 1. Load Metadata/Labels
    if mode == "train":
        df_meta = pd.read_csv(Config.METADATA_TRAIN)
        df_val = pd.read_csv(Config.METADATA_VAL)
        df_labels = pd.concat([df_meta, df_val], ignore_index=True)
        tracking_path = Config.TRAIN_TRACKING
        helmets_path = Config.TRAIN_HELMETS
    else:
        df_labels = pd.read_csv(Config.METADATA_TEST)
        tracking_path = Config.TEST_TRACKING
        helmets_path = Config.TEST_HELMETS

    # 2. Load and Prepare Tracking
    print("Loading tracking data...")
    df_tracking = pd.read_csv(tracking_path, usecols=Config.TRACKING_COLS)

    # Filter tracking to relevant game_plays
    relevant_gps = df_labels["game_play"].unique()
    df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_gps)].copy()

    print("Generating windowed kinematic features...")
    df_tracking_wide = prepare_tracking_data(df_tracking)

    # 3. Merge Tracking to Labels (P1 and P2)
    print("Merging tracking data...")
    # P1
    df_merged = df_labels.merge(
        df_tracking_wide,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(columns=["nfl_player_id"])

    # Rename P1 columns
    p1_cols = [
        c
        for c in df_tracking_wide.columns
        if c not in ["game_play", "nfl_player_id", "step"]
    ]
    df_merged = df_merged.rename(columns={c: f"p1_{c}" for c in p1_cols})

    # P2 (Handle Ground)
    # Create a temporary P2 ID column that is numeric, G becomes NaN
    df_merged["nfl_player_id_2_num"] = pd.to_numeric(
        df_merged["nfl_player_id_2"], errors="coerce"
    )

    df_merged = df_merged.merge(
        df_tracking_wide,
        left_on=["game_play", "nfl_player_id_2_num", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
        suffixes=("", "_p2"),
    ).drop(columns=["nfl_player_id"])

    # Rename P2 columns
    df_merged = df_merged.rename(columns={c: f"p2_{c}" for c in p1_cols})

    # 4. Ground Imputation & Relative Physics
    print("Computing relative physics...")
    is_ground = df_merged["nfl_player_id_2"] == "G"

    # Loop through lags to compute relative features per lag
    # This maintains the temporal structure for the Gated Network
    base_feats = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "direction",
        "orientation",
        "distance",
        "sa",
    ]

    for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        suffix = f"_lag_{lag}"

        # Ground Imputation for P2
        # If Ground: P2 pos = P1 pos, P2 vel/acc = 0
        for feat in base_feats:
            p1_col = f"p1_{feat}{suffix}"
            p2_col = f"p2_{feat}{suffix}"

            if feat in ["x_position", "y_position"]:
                # P2 pos = P1 pos where ground
                df_merged.loc[is_ground, p2_col] = df_merged.loc[is_ground, p1_col]
            else:
                # P2 dynamics = 0 where ground
                df_merged.loc[is_ground, p2_col] = 0.0

            # Fill remaining NaNs (missing tracking) with 0
            df_merged[p1_col] = df_merged[p1_col].fillna(0)
            df_merged[p2_col] = df_merged[p2_col].fillna(0)

        # Relative Features
        dx = df_merged[f"p1_x_position{suffix}"] - df_merged[f"p2_x_position{suffix}"]
        dy = df_merged[f"p1_y_position{suffix}"] - df_merged[f"p2_y_position{suffix}"]
        dist = np.sqrt(dx**2 + dy**2)

        # Log distance
        df_merged[f"log_dist{suffix}"] = np.log1p(dist)

        # Relative Speed
        df_merged[f"rel_speed{suffix}"] = np.abs(
            df_merged[f"p1_speed{suffix}"] - df_merged[f"p2_speed{suffix}"]
        )

        # Relative Angle (Shortest Arc)
        # We use direction for motion vector
        df_merged[f"rel_angle{suffix}"] = angular_diff(
            df_merged[f"p1_direction{suffix}"], df_merged[f"p2_direction{suffix}"]
        )

    # 5. Visual Features (Max Pooling)
    print("Processing visual features...")
    df_helmets = pd.read_csv(helmets_path)
    df_helmets = df_helmets[df_helmets["game_play"].isin(relevant_gps)].copy()

    # Calculate area for max pooling
    df_helmets["area"] = df_helmets["width"] * df_helmets["height"]

    # Select best view per player/frame
    # Group by game_play, nfl_player_id, frame (frame is roughly step * 6 + start_frame, but we use map)
    # Actually, helmets are by frame. We need to map step to frame or join by nearest.
    # The prompt says labels are 10Hz, video 59.94Hz.
    # Simple approximation: frame = step * 6 + 300 (snap at 300).
    # Step 0 is snap.
    # Let's verify mapping. "step: ... starting at 0 at the moment of the play starting... incrementing by 1 every 0.1 seconds"
    # "The moment of snap occurs 5 seconds into the video." -> Frame 300 (approx 5 * 59.94).
    # So step 0 -> frame 300. Step 1 -> Frame 306.

    df_helmets["step_approx"] = ((df_helmets["frame"] - 300) / 6).round().astype(int)

    # Filter to relevant steps
    # Max pooling: sort by area desc, drop duplicates
    df_helmets_best = df_helmets.sort_values("area", ascending=False).drop_duplicates(
        subset=["game_play", "nfl_player_id", "step_approx"]
    )

    # Select features
    vis_cols = ["left", "width", "top", "height"]
    df_vis = df_helmets_best[
        ["game_play", "nfl_player_id", "step_approx"] + vis_cols
    ].copy()

    # Merge Visuals to P1
    df_merged = df_merged.merge(
        df_vis,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step_approx"],
        how="left",
    ).drop(columns=["nfl_player_id", "step_approx"])

    # Rename Visual Cols
    df_merged = df_merged.rename(columns={c: f"v_{c}" for c in vis_cols})

    # Fill missing visuals with 0
    for c in vis_cols:
        df_merged[f"v_{c}"] = df_merged[f"v_{c}"].fillna(0)

    # Standardize Visuals (simple normalization by image size approx 1280x720)
    df_merged["v_left"] /= 1280.0
    df_merged["v_width"] /= 1280.0
    df_merged["v_top"] /= 720.0
    df_merged["v_height"] /= 720.0

    # 6. Final Cleanup
    # Drop intermediate columns
    drop_cols = [
        "nfl_player_id_2_num",
        "datetime",
        "path_endzone",
        "path_sideline",
        "path_all29",
    ]
    df_merged = df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns])

    # Save to cache
    print(f"Saving processed data to {cache_file}...")
    df_merged.to_parquet(cache_file)

    return df_merged


# ==========================================
# Dataset & Model
# ==========================================


class NFLContactDataset(Dataset):
    def __init__(self, df, feature_cols, visual_cols, target_col=None):
        self.features = df[feature_cols].values.astype(np.float32)
        self.visuals = df[visual_cols].values.astype(np.float32)
        self.targets = (
            df[target_col].values.astype(np.float32)
            if target_col in df.columns
            else None
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x_kin = self.features[idx]
        x_vis = self.visuals[idx]

        if self.targets is not None:
            y = self.targets[idx]
            return x_kin, x_vis, y
        return x_kin, x_vis


class GatedBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # GLU: (W1 x) * sigmoid(W2 x)
        gate = torch.sigmoid(self.linear2(x))
        out = self.linear1(x) * gate
        out = self.dropout(out)
        return self.norm(out)


class GRVCNet(nn.Module):
    def __init__(self, kin_input_dim, vis_input_dim, config):
        super().__init__()

        # Kinematic Stream (Gated Backbone)
        self.kin_encoder = nn.Sequential(
            GatedBlock(kin_input_dim, config.HIDDEN_DIM, config.DROPOUT),
            GatedBlock(config.HIDDEN_DIM, config.HIDDEN_DIM, config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 1),
        )

        # Visual Stream (Shallow MLP)
        self.vis_encoder = nn.Sequential(
            nn.Linear(vis_input_dim, config.VISUAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.VISUAL_HIDDEN_DIM, 1),
        )

        # Fusion Weight (Learnable)
        self.fusion_lambda = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_kin, x_vis):
        kin_logit = self.kin_encoder(x_kin)
        vis_logit = self.vis_encoder(x_vis)

        # Residual Fusion
        final_logit = kin_logit + self.fusion_lambda * vis_logit
        return final_logit.squeeze()


# ==========================================
# Training Utilities
# ==========================================


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x_kin, x_vis, y in loader:
        x_kin, x_vis, y = x_kin.to(device), x_vis.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x_kin, x_vis)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_kin, x_vis, y in loader:
            x_kin, x_vis, y = x_kin.to(device), x_vis.to(device), y.to(device)
            logits = model(x_kin, x_vis)
            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate MCC at default 0.5 threshold for monitoring
    mcc = matthews_corrcoef(all_targets, (all_preds > 0.5).astype(int))

    return total_loss / len(loader), mcc, all_preds, all_targets


def run_pipeline():
    # 1. Load Data
    df_train_all = process_dataset(mode="train", load_cached_data=True)
    df_test = process_dataset(mode="test", load_cached_data=True)

    # 2. Split Train/Val based on Metadata
    # We need to reload metadata to get the split IDs because process_dataset merges them
    meta_val = pd.read_csv(Config.METADATA_VAL)
    val_gps = meta_val["game_play"].unique()

    is_val = df_train_all["game_play"].isin(val_gps)
    df_train = df_train_all[~is_val].copy()
    df_val = df_train_all[is_val].copy()

    print(f"Train size: {len(df_train)}, Val size: {len(df_val)}")

    # 3. Identify Feature Columns
    # All columns ending in lag_X or v_X
    # Kinematic: p1_*, p2_*, log_dist*, rel_speed*, rel_angle*
    # Visual: v_*

    all_cols = df_train.columns
    vis_cols = [c for c in all_cols if c.startswith("v_")]
    exclude = [
        "contact_id",
        "game_play",
        "contact",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
    ] + vis_cols
    kin_cols = [c for c in all_cols if c not in exclude]

    print(f"Num Kinematic Features: {len(kin_cols)}")
    print(f"Num Visual Features: {len(vis_cols)}")

    # 4. Scale Data
    scaler = StandardScaler()
    df_train[kin_cols] = scaler.fit_transform(df_train[kin_cols])
    df_val[kin_cols] = scaler.transform(df_val[kin_cols])
    df_test[kin_cols] = scaler.transform(df_test[kin_cols])

    # Save scaler
    joblib.dump(scaler, Config.SCALER_PATH)

    # 5. Create Datasets/Loaders
    train_ds = NFLContactDataset(df_train, kin_cols, vis_cols, "contact")
    val_ds = NFLContactDataset(df_val, kin_cols, vis_cols, "contact")
    test_ds = NFLContactDataset(df_test, kin_cols, vis_cols, None)

    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4
    )

    # 6. Model & Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRVCNet(len(kin_cols), len(vis_cols), Config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    best_mcc = -1
    patience_counter = 0
    best_model_state = None

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mcc, _, _ = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}, Val MCC {val_mcc:.6f}"
        )

        if val_mcc > best_mcc:
            best_mcc = val_mcc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # 7. Threshold Optimization
    print("Optimizing threshold...")
    _, _, val_probs, val_targets = validate(model, val_loader, criterion, device)

    best_thresh = 0.5
    best_val_mcc = -1
    thresholds = np.arange(0.1, 0.9, 0.01)

    for t in thresholds:
        mcc = matthews_corrcoef(val_targets, (val_probs > t).astype(int))
        if mcc > best_val_mcc:
            best_val_mcc = mcc
            best_thresh = t

    print(f"Best Threshold: {best_thresh:.4f}, Best MCC: {best_val_mcc:.6f}")

    # 8. Inference
    print("Generating predictions...")
    model.eval()
    test_preds = []

    with torch.no_grad():
        for x_kin, x_vis in test_loader:
            x_kin, x_vis = x_kin.to(device), x_vis.to(device)
            logits = model(x_kin, x_vis)
            probs = torch.sigmoid(logits)
            test_preds.append(probs.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    binary_preds = (test_preds > best_thresh).astype(int)

    # 9. Submission
    df_sub = pd.DataFrame(
        {"contact_id": df_test["contact_id"], "contact": binary_preds}
    )

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# Note: The pipeline is not executed automatically.
# To run: run_pipeline()
