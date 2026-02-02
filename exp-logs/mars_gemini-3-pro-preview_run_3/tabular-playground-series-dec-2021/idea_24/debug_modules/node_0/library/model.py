import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything, get_device

# --------------------------------------------------------------------------
# 1. Model Architecture
# --------------------------------------------------------------------------


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l
    where (.) denotes dot product (sum over features) resulting in a scalar,
    and (*) denotes scalar broadcasting over the vector x_0.
    """

    def __init__(self, input_dim):
        super(VectorCrossLayer, self).__init__()
        self.input_dim = input_dim
        # Parameters w and b are vectors of size input_dim
        self.w = nn.Parameter(torch.randn(input_dim))
        self.b = nn.Parameter(torch.zeros(input_dim))

        # Initialize w to be small to start near identity behavior
        nn.init.xavier_uniform_(self.w.unsqueeze(0))

    def forward(self, x0, xl):
        # x0: [Batch, Dim] - Original input
        # xl: [Batch, Dim] - Input from previous layer

        # Compute scalar dot product per sample: (xl * w).sum()
        # (xl * self.w) is element-wise [Batch, Dim]
        # .sum(dim=1) results in [Batch, 1]
        dot_prod = (xl * self.w).sum(dim=1, keepdim=True)

        # Apply formula: x0 * scalar + b + xl
        out = x0 * dot_prod + self.b + xl
        return out


class PreActResNetBlock(nn.Module):
    """
    Pre-Activation ResNet Block.
    Topology: BN -> ReLU -> Dropout -> Linear
    Residual: x + f(x)
    """

    def __init__(self, hidden_dim, dropout_rate):
        super(PreActResNetBlock, self).__init__()
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

        # Initialize linear layer
        nn.init.kaiming_normal_(self.linear.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # Pre-activation path
        out = self.bn(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear(out)

        # Residual connection
        return x + out


class ParallelDCNResNet(nn.Module):
    """
    Deep Parallel Vector-DCN-ResNet (Pre-Activation Variant).

    Structure:
    - Input Layer
    - Branch 1: Stack of VectorCrossLayers (DCN)
    - Branch 2: Projection -> Stack of PreActResNetBlocks (ResNet)
    - Concatenation
    - Final Classification Head
    """

    def __init__(self, input_dim, hidden_dim, num_blocks, num_classes, dropout_rate):
        super(ParallelDCNResNet, self).__init__()

        # --- Branch 1: DCN ---
        # Stack of VectorCrossLayers
        self.dcn_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(num_blocks)]
        )

        # --- Branch 2: ResNet ---
        # Projection to hidden dimension
        self.project = nn.Linear(input_dim, hidden_dim)

        # Stack of Pre-Activation ResNet Blocks
        self.resnet_blocks = nn.ModuleList(
            [PreActResNetBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        # --- Combination Head ---
        concat_dim = input_dim + hidden_dim
        self.head = nn.Linear(concat_dim, num_classes)

        # Initialization
        nn.init.kaiming_normal_(
            self.project.weight, mode="fan_in", nonlinearity="linear"
        )
        nn.init.zeros_(self.project.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: [Batch, Input_Dim]

        # Branch 1: DCN
        # x0 is fixed as the original input for Cross Net
        x_dcn = x
        x0 = x
        for layer in self.dcn_layers:
            x_dcn = layer(x0, x_dcn)

        # Branch 2: ResNet
        x_res = self.project(x)
        for block in self.resnet_blocks:
            x_res = block(x_res)

        # Concatenate outputs
        combined = torch.cat([x_dcn, x_res], dim=1)

        # Classification
        logits = self.head(combined)
        return logits


# --------------------------------------------------------------------------
# 2. Data Processing
# --------------------------------------------------------------------------


def feature_engineering(df):
    """
    Applies Augmented Physics-Informed Engineering.
    Generates new features while preserving raw signals.
    """
    # Work on a copy
    df = df.copy()

    # 1. Cyclical Augmentation for Aspect
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    # sqrt(H^2 + V^2)
    h_dist = df["Horizontal_Distance_To_Hydrology"]
    v_dist = df["Vertical_Distance_To_Hydrology"]
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(h_dist**2 + v_dist**2)

    # 3. Directional Preservation (Hydrology Elevation)
    # Elevation - Vertical Distance
    df["Hydrology_Elevation"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # 4. Global Context (Mean Distance to Amenities)
    # Mean of distances to Hydrology, Roadways, Fire Points
    amenities = [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]
    df["Mean_Distance_To_Amenities"] = df[amenities].mean(axis=1)

    return df


def process_data(load_cached_data=True):
    """
    Loads raw parquet files, applies feature engineering and scaling,
    and caches the processed numpy arrays.

    Returns:
        X_train, y_train, X_val, y_val, X_test, test_ids
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check cache existence
    cache_exists = all(os.path.exists(f) for f in files.values())

    if load_cached_data and cache_exists:
        print("Loading cached processed data...")
        X_train = np.load(files["X_train"])
        y_train = np.load(files["y_train"])
        X_val = np.load(files["X_val"])
        y_val = np.load(files["y_val"])
        X_test = np.load(files["X_test"])
        test_ids = np.load(files["test_ids"])
        return X_train, y_train, X_val, y_val, X_test, test_ids

    print("Processing data from scratch...")

    # Load Metadata
    df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)
    df_test = pd.read_parquet(Config.TEST_DATA_PATH)

    # Extract Test IDs
    test_ids = df_test[Config.ID_COL].values

    # Apply Feature Engineering
    print("Applying feature engineering...")
    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)

    # Define columns
    cont_cols = Config.CONTINUOUS_FEATURES
    bin_cols = Config.BINARY_FEATURES

    # Validate columns
    for c in cont_cols + bin_cols:
        if c not in df_train.columns:
            raise ValueError(f"Expected column {c} not found in dataframe.")

    # Standardization (Continuous Features Only)
    print("Standardizing continuous features...")
    scaler = StandardScaler()
    # Fit on Train
    X_train_cont = scaler.fit_transform(df_train[cont_cols].values.astype(np.float32))
    # Transform Val and Test
    X_val_cont = scaler.transform(df_val[cont_cols].values.astype(np.float32))
    X_test_cont = scaler.transform(df_test[cont_cols].values.astype(np.float32))

    # Extract Binary Features (No scaling)
    X_train_bin = df_train[bin_cols].values.astype(np.float32)
    X_val_bin = df_val[bin_cols].values.astype(np.float32)
    X_test_bin = df_test[bin_cols].values.astype(np.float32)

    # Concatenate Features
    X_train = np.hstack([X_train_cont, X_train_bin])
    X_val = np.hstack([X_val_cont, X_val_bin])
    X_test = np.hstack([X_test_cont, X_test_bin])

    # Process Targets (Map 1-7 to 0-6)
    y_train = (df_train[Config.TARGET_COL].values - 1).astype(np.int64)
    y_val = (df_val[Config.TARGET_COL].values - 1).astype(np.int64)

    # Save to Cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(files["X_train"], X_train)
    np.save(files["y_train"], y_train)
    np.save(files["X_val"], X_val)
    np.save(files["y_val"], y_val)
    np.save(files["X_test"], X_test)
    np.save(files["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids


# --------------------------------------------------------------------------
# 3. Training & Inference
# --------------------------------------------------------------------------


def train_model():
    """
    Trains the ParallelDCNResNet model.
    Returns:
        model: The best trained model (loaded with best weights).
        X_test: Processed test features.
        test_ids: IDs for the test set.
    """
    seed_everything(Config.SEED)
    device = get_device()

    # 1. Load Data
    X_train, y_train, X_val, y_val, X_test, test_ids = process_data(
        load_cached_data=True
    )

    # 2. Create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Initialize Model
    input_dim = X_train.shape[1]
    model = ParallelDCNResNet(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_BLOCKS,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT,
    ).to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_acc = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Scheduler Step
        scheduler.step(val_acc)

        # Early Stopping & Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save intermediate best model
            torch.save(
                best_model_state, os.path.join(Config.WORKING_DIR, "best_model.pth")
            )
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training Complete. Best Validation Accuracy: {best_acc:.6f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, X_test, test_ids


def generate_submission(model, X_test, test_ids):
    """
    Generates predictions for the test set and saves to CSV.
    """
    device = get_device()
    model.eval()

    test_dataset = TensorDataset(torch.tensor(X_test))
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4
    )

    predictions = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for (X_batch,) in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())

    # Map predictions back to 1-7 range
    final_preds = np.array(predictions) + 1

    # Create Submission DataFrame
    submission = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: final_preds})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


def run():
    """
    Main execution function.
    """
    model, X_test, test_ids = train_model()
    generate_submission(model, X_test, test_ids)
