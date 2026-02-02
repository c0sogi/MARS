import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from library.config import Config


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(Config.SEED)

# --------------------------------------------------------------------------
# Model Architecture
# --------------------------------------------------------------------------


class CrossLayer(nn.Module):
    """
    DCN-V2 Cross Layer (Vector-based): x_{l+1} = x_0 * (x_l^T w) + b + x_l
    Explicitly models feature interactions using O(d) parameters to prevent overfitting.
    (Cite Lesson 00027)
    """

    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.input_dim = input_dim
        # Vector weight w: [d]
        self.weight = nn.Parameter(torch.Tensor(input_dim))
        # Bias b: [d]
        self.bias = nn.Parameter(torch.Tensor(input_dim))

        # Initialize
        nn.init.normal_(self.weight, std=0.01)
        nn.init.zeros_(self.bias)

    def forward(self, x_l, x_0):
        # x_l: [batch_size, input_dim]
        # x_0: [batch_size, input_dim]

        # Compute dot product (x_l^T w) -> Scalar per sample
        # x_l * weight -> [batch, d] -> sum(dim=1) -> [batch, 1]
        dot = (x_l * self.weight).sum(dim=1, keepdim=True)

        # x_{l+1} = x_0 * dot + b + x_l
        out = x_0 * dot + self.bias + x_l
        return out


class ResNetBlock(nn.Module):
    """
    Residual Block: x + Linear2(Dropout(ReLU(BN(Linear1(x)))))
    """

    def __init__(self, width, dropout_rate):
        super(ResNetBlock, self).__init__()
        self.linear1 = nn.Linear(width, width)
        self.bn = nn.BatchNorm1d(width)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(width, width)

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return identity + out


class ParallelDCNResNet(nn.Module):
    """
    Hybrid Architecture:
    1. Deep Branch (ResNet): Captures implicit non-linear patterns.
    2. Cross Branch (DCN): Captures explicit bounded-degree feature interactions.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        resnet_blocks=Config.RESNET_BLOCKS,
        resnet_width=Config.RESNET_WIDTH,
        resnet_dropout=Config.RESNET_DROPOUT,
        dcn_layers=Config.DCN_LAYERS,
    ):
        super(ParallelDCNResNet, self).__init__()

        # --- ResNet Branch ---
        # Projection from input dimension to ResNet width
        self.resnet_projection = nn.Linear(input_dim, resnet_width)

        # Stack of Residual Blocks
        self.resnet_blocks = nn.ModuleList(
            [ResNetBlock(resnet_width, resnet_dropout) for _ in range(resnet_blocks)]
        )

        # --- Cross Branch ---
        # Stack of Cross Layers
        self.cross_layers = nn.ModuleList(
            [CrossLayer(input_dim) for _ in range(dcn_layers)]
        )

        # --- Combination Head ---
        # Concatenate ResNet output (resnet_width) and DCN output (input_dim)
        concat_dim = resnet_width + input_dim
        self.head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, input_dim]

        # 1. ResNet Branch
        res_out = self.resnet_projection(x)
        for block in self.resnet_blocks:
            res_out = block(res_out)

        # 2. Cross Branch
        # DCN requires x_0 (original input) at each step
        dcn_out = x
        x_0 = x
        for layer in self.cross_layers:
            dcn_out = layer(dcn_out, x_0)

        # 3. Fusion
        combined = torch.cat([res_out, dcn_out], dim=1)
        logits = self.head(combined)

        return logits


# --------------------------------------------------------------------------
# Data Processing & Feature Engineering
# --------------------------------------------------------------------------


def apply_feature_engineering(df):
    """
    Applies Physics-Informed Feature Engineering.
    """
    # Ensure inputs are float for calculation
    df["Elevation"] = df["Elevation"].astype(float)
    df["Vertical_Distance_To_Hydrology"] = df["Vertical_Distance_To_Hydrology"].astype(
        float
    )
    df["Horizontal_Distance_To_Hydrology"] = df[
        "Horizontal_Distance_To_Hydrology"
    ].astype(float)
    df["Horizontal_Distance_To_Roadways"] = df[
        "Horizontal_Distance_To_Roadways"
    ].astype(float)
    df["Horizontal_Distance_To_Fire_Points"] = df[
        "Horizontal_Distance_To_Fire_Points"
    ].astype(float)

    # 1. Geometric Magnitude: Euclidean Distance to Hydrology
    df["Hydro_Euclidean"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 2. Directional Preservation: Absolute Hydrology Elevation
    # Elevation of the water source relative to sea level (approx)
    df["Hydro_Elevation"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # 3. Global Remoteness: Mean Distance to Amenities
    df["Mean_Amenity_Dist"] = (
        df["Hydro_Euclidean"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ) / 3.0

    return df


def get_data(load_cached_data=True, debug_samples=Config.MAX_DEBUG_SAMPLES):
    """
    Loads data, performs feature engineering, and caches results.
    Strictly follows the caching logic requirement.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Paths
    cache_train_x = Config.CACHE_TRAIN_X
    cache_train_y = Config.CACHE_TRAIN_Y
    cache_val_x = Config.CACHE_VAL_X
    cache_val_y = Config.CACHE_VAL_Y
    cache_test_x = Config.CACHE_TEST_X
    cache_test_ids = Config.CACHE_TEST_IDS

    # Check if all caches exist
    caches_exist = (
        os.path.exists(cache_train_x)
        and os.path.exists(cache_train_y)
        and os.path.exists(cache_val_x)
        and os.path.exists(cache_val_y)
        and os.path.exists(cache_test_x)
        and os.path.exists(cache_test_ids)
    )

    if load_cached_data and caches_exist:
        print("Loading cached data from .npy files...")
        X_train = np.load(cache_train_x)
        y_train = np.load(cache_train_y)
        X_val = np.load(cache_val_x)
        y_val = np.load(cache_val_y)
        X_test = np.load(cache_test_x)
        test_ids = np.load(cache_test_ids)
        return X_train, y_train, X_val, y_val, X_test, test_ids

    print("Processing data from scratch...")

    # Load Parquet
    df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)
    df_test = pd.read_parquet(Config.TEST_DATA_PATH)

    # Debugging subsample
    if debug_samples is not None:
        print(f"Subsampling {debug_samples} rows for debugging...")
        df_train = df_train.iloc[:debug_samples]
        df_val = df_val.iloc[:debug_samples]
        # Keep test full usually, but for debug logic we might leave it.
        # But prediction requires full test. Let's not subsample test for safety unless strictly debugging flow.

    # Feature Engineering
    print("Applying Physics-Informed Feature Engineering...")
    df_train = apply_feature_engineering(df_train)
    df_val = apply_feature_engineering(df_val)
    df_test = apply_feature_engineering(df_test)

    # Separate Target and IDs
    y_train = df_train[Config.TARGET_COL].values
    y_val = df_val[Config.TARGET_COL].values
    test_ids = df_test[Config.ID_COL].values

    # Drop ID and Target from features
    drop_cols_train = [Config.TARGET_COL]
    if Config.ID_COL in df_train.columns:
        drop_cols_train.append(Config.ID_COL)

    drop_cols_val = [Config.TARGET_COL]
    if Config.ID_COL in df_val.columns:
        drop_cols_val.append(Config.ID_COL)

    drop_cols_test = []
    if Config.ID_COL in df_test.columns:
        drop_cols_test.append(Config.ID_COL)

    X_train_df = df_train.drop(columns=drop_cols_train)
    X_val_df = df_val.drop(columns=drop_cols_val)
    X_test_df = df_test.drop(columns=drop_cols_test)

    # Identify Continuous vs Binary columns
    # Binary columns start with prefixes in Config.BINARY_PREFIXES
    all_cols = X_train_df.columns.tolist()
    binary_cols = [
        c for c in all_cols if any(c.startswith(p) for p in Config.BINARY_PREFIXES)
    ]
    continuous_cols = [c for c in all_cols if c not in binary_cols]

    print(
        f"Features: {len(all_cols)} Total. {len(continuous_cols)} Continuous, {len(binary_cols)} Binary."
    )

    # Standardization (Fit on Train, Transform All)
    # Only standardize continuous columns
    scaler = StandardScaler()

    X_train_cont = scaler.fit_transform(X_train_df[continuous_cols].values)
    X_val_cont = scaler.transform(X_val_df[continuous_cols].values)
    X_test_cont = scaler.transform(X_test_df[continuous_cols].values)

    # Concatenate back with binary columns (kept as is)
    X_train = np.hstack([X_train_cont, X_train_df[binary_cols].values])
    X_val = np.hstack([X_val_cont, X_val_df[binary_cols].values])
    X_test = np.hstack([X_test_cont, X_test_df[binary_cols].values])

    # Cast to float32 for PyTorch
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Encode Targets (Shift to 0-indexed if necessary)
    # Cover_Type is 1-7. PyTorch CrossEntropy expects 0-(C-1).
    # We will use a LabelEncoder logic or just subtract min if contiguous.
    # The dataset analysis showed classes 1, 2, 3, 4, 6, 7. Class 5 is missing?
    # To be safe, we map classes to 0..N-1.
    unique_classes = sorted(np.unique(np.concatenate([y_train, y_val])))
    class_map = {c: i for i, c in enumerate(unique_classes)}
    inverse_class_map = {i: c for c, i in class_map.items()}

    # Save the class map logic? We need it for submission.
    # Since we can't pickle, we'll reconstruct it or just rely on the fact that we know the set.
    # We will assume the caller handles this or we return the map?
    # For simplicity, we'll just subtract 1 if classes are 1-7, but since 5 is missing,
    # using a direct map is safer for Softmax.

    y_train_enc = np.array([class_map[c] for c in y_train], dtype=np.int64)
    y_val_enc = np.array([class_map[c] for c in y_val], dtype=np.int64)

    # Cache the processed data
    # Note: We are caching the *encoded* y. We need to remember to decode for submission.
    # But wait, we can't easily cache the map in .npy.
    # Strategy: We will assume standard 1-7 mapping for now, or just re-compute map on load?
    # For robustness, let's cache the raw y and re-encode in the training loop.
    # Re-saving raw y to cache.

    print("Caching data...")
    np.save(cache_train_x, X_train)
    np.save(cache_train_y, y_train)  # Save raw
    np.save(cache_val_x, X_val)
    np.save(cache_val_y, y_val)  # Save raw
    np.save(cache_test_x, X_test)
    np.save(cache_test_ids, test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids


# --------------------------------------------------------------------------
# Training & Inference Pipeline
# --------------------------------------------------------------------------


def run_training_pipeline(
    epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached=True
):
    """
    Orchestrates the full pipeline: Data Loading -> Training -> Inference -> Submission.
    """
    # 1. Load Data
    X_train, y_train_raw, X_val, y_val_raw, X_test, test_ids = get_data(
        load_cached_data=load_cached
    )

    # Encode Targets
    unique_classes = sorted(np.unique(np.concatenate([y_train_raw, y_val_raw])))
    class_map = {c: i for i, c in enumerate(unique_classes)}
    inverse_class_map = {i: c for c, i in class_map.items()}
    num_classes = len(unique_classes)

    y_train = np.array([class_map[c] for c in y_train_raw], dtype=np.int64)
    y_val = np.array([class_map[c] for c in y_val_raw], dtype=np.int64)

    print(
        f"Data Loaded. Train: {X_train.shape}, Val: {X_val.shape}, Classes: {num_classes}"
    )

    # Create DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model
    device = Config.DEVICE
    input_dim = X_train.shape[1]

    model = ParallelDCNResNet(input_dim=input_dim, num_classes=num_classes).to(device)

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

    # 3. Training Loop with Early Stopping
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print("Starting Training...")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validate
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        # Update Scheduler
        scheduler.step(epoch_val_acc)

        # Logging
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {epoch_train_loss:.6f}, Train Acc: {epoch_train_acc:.6f}, "
            f"Val Loss: {epoch_val_loss:.6f}"
        )
        print(f"Validation Accuracy: {epoch_val_acc}")  # Full precision as requested

        # Early Stopping Check
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save checkpoint
            torch.save(best_model_wts, Config.MODEL_CHECKPOINT_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best weights
    print(f"Loading best model weights with Validation Accuracy: {best_val_acc}")
    model.load_state_dict(best_model_wts)

    # 4. Inference on Test Set
    print("Generating predictions on Test Set...")
    test_dataset = TensorDataset(torch.from_numpy(X_test))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    # Map back to original class labels
    final_preds = [inverse_class_map[p] for p in predictions]

    # 5. Save Submission
    submission_df = pd.DataFrame(
        {Config.ID_COL: test_ids, Config.TARGET_COL: final_preds}
    )

    # Ensure ID format matches sample (int)
    submission_df[Config.ID_COL] = submission_df[Config.ID_COL].astype(int)

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")

    return best_val_acc
