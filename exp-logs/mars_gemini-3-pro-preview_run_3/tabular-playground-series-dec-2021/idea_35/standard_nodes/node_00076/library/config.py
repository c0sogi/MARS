import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Config:
    # ==============================
    # File Paths
    # ==============================
    METADATA_DIR = "./metadata"
    TRAIN_DATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA = os.path.join(METADATA_DIR, "test.parquet")

    WORKING_DIR = "./working/idea_35"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==============================
    # Hyperparameters
    # ==============================
    SEED = 42
    BATCH_SIZE = 4096
    EPOCHS = 60
    LR = 1e-3
    WEIGHT_DECAY = 1e-2  # Decoupled weight decay for AdamW
    DROPOUT = 0.2

    # Scheduler
    SCHEDULER_FACTOR = 0.1
    SCHEDULER_PATIENCE = 5

    # Model Architecture
    HIDDEN_DIM = 512
    RESNET_BLOCKS = 4
    DCN_LAYERS = 3

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


# =============================================================================
# Feature Engineering & Data Processing
# =============================================================================


def feature_engineering(df):
    """
    Applies Augmented Physics-Informed Engineering.
    """
    df = df.copy()

    # 1. Cyclical Augmentation (Keep raw Aspect as well)
    # Aspect is in degrees 0-360
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Geometric Magnitude: Euclidean Distance to Hydrology
    # sqrt(H^2 + V^2)
    df["Hydrology_Euclidean"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation: Absolute Hydrology Elevation
    # Elevation - Vertical_Distance
    df["Hydrology_Elevation_Abs"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    # 4. Global Context: Mean Distance to Amenities
    # Mean of distances to Hydrology, Roadways, Fire Points
    df["Amenities_Mean_Dist"] = df[
        [
            "Horizontal_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points",
        ]
    ].mean(axis=1)

    return df


def process_data(load_cached_data=True):
    """
    Loads, processes, and caches data.
    Returns:
        X_train, y_train, X_val, y_val, X_test, test_ids, num_features, num_classes, label_encoder
    """
    cache_path = os.path.join(Config.CACHE_DIR, "processed_data.npy")
    meta_path = os.path.join(Config.CACHE_DIR, "metadata.npy")  # Stores scalar info

    if load_cached_data and os.path.exists(cache_path) and os.path.exists(meta_path):
        print(f"Loading cached data from {Config.CACHE_DIR}...")
        data_dict = np.load(cache_path, allow_pickle=True).item()
        meta_dict = np.load(meta_path, allow_pickle=True).item()
        return (
            data_dict["X_train"],
            data_dict["y_train"],
            data_dict["X_val"],
            data_dict["y_val"],
            data_dict["X_test"],
            data_dict["test_ids"],
            meta_dict["num_features"],
            meta_dict["num_classes"],
            meta_dict["label_encoder"],
        )

    print("Processing data from scratch...")

    # Load raw parquet files
    train_df = pd.read_parquet(Config.TRAIN_DATA)
    val_df = pd.read_parquet(Config.VAL_DATA)
    test_df = pd.read_parquet(Config.TEST_DATA)

    # Extract IDs for test set
    test_ids = test_df["Id"].values

    # Separate Target
    target_col = "Cover_Type"
    y_train_raw = train_df[target_col].values
    y_val_raw = val_df[target_col].values

    # Drop Id and Target from features
    drop_cols = ["Id", target_col]
    X_train_df = train_df.drop(columns=drop_cols, errors="ignore")
    X_val_df = val_df.drop(columns=drop_cols, errors="ignore")
    X_test_df = test_df.drop(
        columns=["Id"], errors="ignore"
    )  # Test doesn't have target

    # Apply Feature Engineering
    X_train_df = feature_engineering(X_train_df)
    X_val_df = feature_engineering(X_val_df)
    X_test_df = feature_engineering(X_test_df)

    # Identify Column Types
    # Binary columns: Soil_Type* and Wilderness_Area*
    binary_cols = [
        c
        for c in X_train_df.columns
        if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
    ]
    continuous_cols = [c for c in X_train_df.columns if c not in binary_cols]

    # Preprocessing: Standardize Continuous, Keep Binary as is
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(X_train_df[continuous_cols])
    X_val_cont = scaler.transform(X_val_df[continuous_cols])
    X_test_cont = scaler.transform(X_test_df[continuous_cols])

    # Concatenate back
    X_train = np.hstack([X_train_cont, X_train_df[binary_cols].values])
    X_val = np.hstack([X_val_cont, X_val_df[binary_cols].values])
    X_test = np.hstack([X_test_cont, X_test_df[binary_cols].values])

    # Encode Targets (Map 1,2,3,4,6,7 -> 0,1,2,3,4,5)
    le = LabelEncoder()
    # Fit on all possible classes to be safe, or just train
    # We combine train and val targets to ensure coverage
    all_targets = np.concatenate([y_train_raw, y_val_raw])
    le.fit(all_targets)
    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)

    num_features = X_train.shape[1]
    num_classes = len(le.classes_)

    # Cache results
    data_dict = {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train.astype(np.int64),
        "X_val": X_val.astype(np.float32),
        "y_val": y_val.astype(np.int64),
        "X_test": X_test.astype(np.float32),
        "test_ids": test_ids,
    }
    meta_dict = {
        "num_features": num_features,
        "num_classes": num_classes,
        "label_encoder": le,
    }

    np.save(cache_path, data_dict)
    np.save(meta_path, meta_dict)

    print(f"Data processed and cached to {Config.CACHE_DIR}")

    return (
        data_dict["X_train"],
        data_dict["y_train"],
        data_dict["X_val"],
        data_dict["y_val"],
        data_dict["X_test"],
        data_dict["test_ids"],
        num_features,
        num_classes,
        le,
    )


# =============================================================================
# Model Architecture
# =============================================================================


class VectorDCNLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer.
    x_{l+1} = x_0 * (x_l^T w) + b + x_l
    """

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.w = nn.Parameter(torch.randn(input_dim))
        self.b = nn.Parameter(torch.zeros(input_dim))

        # Init w with near-zero std dev to start as identity
        nn.init.normal_(self.w, mean=0, std=1e-4)

    def forward(self, x0, xl):
        # x0: [batch, dim]
        # xl: [batch, dim]
        # w: [dim]

        # Compute dot product (xl^T w) per sample -> scalar
        # (batch, dim) * (dim) -> (batch, dim) --sum--> (batch)
        dot_prod = torch.sum(xl * self.w, dim=1, keepdim=True)  # [batch, 1]

        # x0 * scalar + b + xl
        out = x0 * dot_prod + self.b + xl
        return out


class ResNetBlock(nn.Module):
    """
    Full Pre-Activation ResNet Block with Swish.
    BN -> Swish -> Dropout -> Linear -> BN -> Swish -> Dropout -> Linear -> Add
    """

    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = nn.SiLU()  # Swish
        self.drop1 = nn.Dropout(dropout_rate)
        self.lin1 = nn.Linear(dim, dim)

        self.bn2 = nn.BatchNorm1d(dim)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.act1(out)
        out = self.drop1(out)
        out = self.lin1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.drop2(out)
        out = self.lin2(out)

        return out + residual


class DeepParallelVectorDCNResNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        # Branch 1: Vector DCN
        # Asymmetric Depth: 3 layers
        self.dcn_layers = nn.ModuleList(
            [VectorDCNLayer(input_dim) for _ in range(Config.DCN_LAYERS)]
        )

        # Branch 2: ResNet Backbone
        # Input Projection to Hidden Dim
        self.resnet_input_proj = nn.Linear(input_dim, Config.HIDDEN_DIM)

        # 4 Blocks
        self.resnet_blocks = nn.Sequential(
            *[
                ResNetBlock(Config.HIDDEN_DIM, Config.DROPOUT)
                for _ in range(Config.RESNET_BLOCKS)
            ]
        )

        # Combination Head: Non-Linear Projection
        # Concat(DCN_Output, ResNet_Output) -> BN -> Swish -> Dropout -> Linear
        combined_dim = input_dim + Config.HIDDEN_DIM

        self.head = nn.Sequential(
            nn.BatchNorm1d(combined_dim),
            nn.SiLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(combined_dim, num_classes),
        )

    def forward(self, x):
        # x: [batch, input_dim]

        # Branch 1: DCN
        x_dcn = x
        x0 = x
        for layer in self.dcn_layers:
            x_dcn = layer(x0, x_dcn)

        # Branch 2: ResNet
        x_res = self.resnet_input_proj(x)
        x_res = self.resnet_blocks(x_res)

        # Combine
        x_combined = torch.cat([x_dcn, x_res], dim=1)

        # Head
        logits = self.head(x_combined)
        return logits


def get_model(input_dim, num_classes):
    return DeepParallelVectorDCNResNet(input_dim, num_classes)
