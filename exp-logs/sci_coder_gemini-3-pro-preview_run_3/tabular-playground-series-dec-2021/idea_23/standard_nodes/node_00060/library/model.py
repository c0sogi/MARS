import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config
import copy


# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)

# -------------------------------------------------------------------------
# Model Components
# -------------------------------------------------------------------------


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer.
    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l
    where (.) denotes dot product (resulting in scalar for each sample),
    and (*) denotes element-wise multiplication (broadcasting scalar).
    """

    def __init__(self, input_dim):
        super(VectorCrossLayer, self).__init__()
        self.input_dim = input_dim
        # w is a weight vector of shape (input_dim, 1)
        self.w = nn.Parameter(torch.randn(input_dim, 1))
        # b is a bias vector of shape (input_dim, )
        self.b = nn.Parameter(torch.zeros(input_dim))

        # Init
        nn.init.xavier_uniform_(self.w)

    def forward(self, x0, xl):
        # xl: (Batch, Dim)
        # w: (Dim, 1)
        # dot_prod: (Batch, 1) = xl @ w
        dot_prod = torch.matmul(xl, self.w)

        # x0 * dot_prod broadcasts (Batch, Dim) * (Batch, 1) -> (Batch, Dim)
        # Add bias and residual
        out = (x0 * dot_prod) + self.b + xl
        return out


class ResNetBlock(nn.Module):
    """
    Dense Residual Block for Tabular Data.
    Structure: Linear -> BN -> ReLU -> Dropout -> Residual
    """

    def __init__(self, hidden_dim, dropout_rate):
        super(ResNetBlock, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + residual


class DeepParallelDCNResNet(nn.Module):
    """
    Hybrid architecture with parallel DCN and ResNet branches.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=512,
        num_res_blocks=4,
        dropout_rate=0.2,
        num_cross_layers=3,
    ):
        super(DeepParallelDCNResNet, self).__init__()

        # Branch 1: Vector DCN
        # Stack of Cross Layers
        self.cross_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(num_cross_layers)]
        )

        # Branch 2: Deep ResNet
        # Projection from input_dim to hidden_dim
        self.resnet_projection = nn.Linear(input_dim, hidden_dim)
        self.resnet_blocks = nn.ModuleList(
            [ResNetBlock(hidden_dim, dropout_rate) for _ in range(num_res_blocks)]
        )

        # Combination Head
        # Concatenate DCN output (input_dim) and ResNet output (hidden_dim)
        concat_dim = input_dim + hidden_dim
        self.head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # Branch 1: DCN
        # x0 is the original input
        x_dcn = x
        for layer in self.cross_layers:
            x_dcn = layer(x, x_dcn)

        # Branch 2: ResNet
        x_res = self.resnet_projection(x)
        for block in self.resnet_blocks:
            x_res = block(x_res)

        # Concatenate
        x_combined = torch.cat([x_dcn, x_res], dim=1)

        # Classification
        logits = self.head(x_combined)
        return logits


# -------------------------------------------------------------------------
# Data Processing & Caching
# -------------------------------------------------------------------------


def feature_engineering(df):
    """
    Applies Augmented Physics-Informed Engineering.
    """
    # 1. Cyclical Augmentation
    df["Aspect_Sin"] = np.sin(df["Aspect"] * np.pi / 180)
    df["Aspect_Cos"] = np.cos(df["Aspect"] * np.pi / 180)

    # 2. Geometric Magnitude
    # Euclidean Distance to Hydrology
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation
    # Absolute Hydrology Elevation
    df["Absolute_Hydrology_Elevation"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    # 4. Global Context
    # Mean Distance to Amenities
    df["Mean_Distance_To_Amenities"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ) / 3.0

    return df


def load_and_process_data(load_cached_data=True, debug=False):
    """
    Loads data, performs feature engineering, scaling, and caching.
    """
    Config.create_directories()

    cache_files = {
        "train_X": os.path.join(Config.CACHE_DIR, "train_X.npy"),
        "train_y": os.path.join(Config.CACHE_DIR, "train_y.npy"),
        "val_X": os.path.join(Config.CACHE_DIR, "val_X.npy"),
        "val_y": os.path.join(Config.CACHE_DIR, "val_y.npy"),
        "test_X": os.path.join(Config.CACHE_DIR, "test_X.npy"),
        "test_ids": os.path.join(Config.CACHE_DIR, "test_ids.npy"),
        "meta": os.path.join(
            Config.CACHE_DIR, "meta.npy"
        ),  # Stores [num_features, classes]
    }

    # Try loading cache
    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading data from cache...")
        X_train = np.load(cache_files["train_X"])
        y_train = np.load(cache_files["train_y"])
        X_val = np.load(cache_files["val_X"])
        y_val = np.load(cache_files["val_y"])
        X_test = np.load(cache_files["test_X"])
        test_ids = np.load(cache_files["test_ids"])
        meta = np.load(cache_files["meta"], allow_pickle=True).item()

        return X_train, y_train, X_val, y_val, X_test, test_ids, meta["classes"]

    print("Processing data from scratch...")

    # Load Parquet
    df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)
    df_test = pd.read_parquet(Config.TEST_DATA_PATH)

    if debug:
        df_train = df_train.iloc[: Config.DEBUG_SAMPLES]
        df_val = df_val.iloc[: Config.DEBUG_SAMPLES]
        df_test = df_test.iloc[: Config.DEBUG_SAMPLES]

    # Feature Engineering
    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)

    # Separate Targets and IDs
    y_train_raw = df_train[Config.TARGET_COL].values
    y_val_raw = df_val[Config.TARGET_COL].values
    test_ids = df_test[Config.ID_COL].values

    # Drop non-feature columns
    drop_cols = [Config.ID_COL, Config.TARGET_COL]
    X_train_df = df_train.drop(columns=drop_cols, errors="ignore")
    X_val_df = df_val.drop(columns=drop_cols, errors="ignore")
    X_test_df = df_test.drop(
        columns=[Config.ID_COL], errors="ignore"
    )  # Test has no target

    # Identify Column Types
    # Continuous: Numerical columns that are not binary
    # Binary: Soil_Type and Wilderness_Area columns
    # Heuristic: If min=0, max=1 and nunique=2, it's binary.
    # However, to be safe and consistent with "Augmented Physics" idea, we manually specify continuous.

    # Base continuous columns
    cont_cols = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
    ]
    # New engineered continuous columns
    new_cols = [
        "Aspect_Sin",
        "Aspect_Cos",
        "Euclidean_Distance_To_Hydrology",
        "Absolute_Hydrology_Elevation",
        "Mean_Distance_To_Amenities",
    ]

    all_cont_cols = [c for c in cont_cols + new_cols if c in X_train_df.columns]

    # Remaining are binary (Soil Types, Wilderness Areas)
    bin_cols = [c for c in X_train_df.columns if c not in all_cont_cols]

    # Preprocessing
    # Standardize Continuous
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(X_train_df[all_cont_cols])
    X_val_cont = scaler.transform(X_val_df[all_cont_cols])
    X_test_cont = scaler.transform(X_test_df[all_cont_cols])

    # Get Binary (already 0/1, just convert to float)
    X_train_bin = X_train_df[bin_cols].values.astype(np.float32)
    X_val_bin = X_val_df[bin_cols].values.astype(np.float32)
    X_test_bin = X_test_df[bin_cols].values.astype(np.float32)

    # Concatenate
    X_train = np.hstack([X_train_cont, X_train_bin])
    X_val = np.hstack([X_val_cont, X_val_bin])
    X_test = np.hstack([X_test_cont, X_test_bin])

    # Encode Targets
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    classes = le.classes_

    # Save to Cache
    np.save(cache_files["train_X"], X_train.astype(np.float32))
    np.save(cache_files["train_y"], y_train.astype(np.int64))
    np.save(cache_files["val_X"], X_val.astype(np.float32))
    np.save(cache_files["val_y"], y_val.astype(np.int64))
    np.save(cache_files["test_X"], X_test.astype(np.float32))
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["meta"], {"classes": classes})

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes


# -------------------------------------------------------------------------
# Training & Inference
# -------------------------------------------------------------------------


def train_and_predict(load_cached_data=True):
    # 1. Load Data
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_and_process_data(
        load_cached_data=load_cached_data, debug=Config.DEBUG
    )

    input_dim = X_train.shape[1]
    num_classes = len(classes)

    print(
        f"Data Loaded. Train: {X_train.shape}, Val: {X_val.shape}, Classes: {num_classes}"
    )

    # 2. Setup Device and DataLoaders
    device = torch.device(Config.DEVICE)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model
    model = DeepParallelDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=Config.HIDDEN_DIM,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = correct / total
        avg_train_loss = train_loss / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Scheduler Step
        scheduler.step(val_acc)

        # Checkpoint & Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Accuracy: {best_val_acc:.6f}")

    # 6. Inference
    print("Generating predictions on test set...")
    model.load_state_dict(best_model_state)
    model.eval()

    test_dataset = TensorDataset(torch.tensor(X_test))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())

    # Inverse transform predictions
    final_preds = classes[all_preds]

    # 7. Submission
    submission = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: final_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
