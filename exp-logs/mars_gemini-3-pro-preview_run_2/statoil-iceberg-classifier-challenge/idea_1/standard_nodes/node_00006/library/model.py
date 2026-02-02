import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from library.config import Config

# Set seeds for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)


class D2N(nn.Module):
    """
    Downsampled Dense Neural Network (D2N).
    A simple MLP that takes flattened downsampled images and incidence angle.
    """

    def __init__(self, input_dim, hidden_units, dropout_rate):
        super(D2N, self).__init__()
        layers = []
        in_features = input_dim

        for units in hidden_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = units

        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def process_image_bands(band_1, band_2, target_size):
    """
    Resizes and flattens image bands.
    """
    # Reshape to 75x75
    b1 = np.array(band_1).reshape(75, 75)
    b2 = np.array(band_2).reshape(75, 75)

    # Resize to target_size (e.g., 32x32)
    # cv2.resize expects (width, height)
    b1_resized = cv2.resize(
        b1, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )
    b2_resized = cv2.resize(
        b2, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )

    # Flatten
    b1_flat = b1_resized.flatten()
    b2_flat = b2_resized.flatten()

    # Concatenate
    return np.concatenate([b1_flat, b2_flat])


def get_data(load_cached_data=True):
    """
    Loads, processes, and returns the data.
    Implements caching mechanism using .npz files.
    """
    cache_file = os.path.join(Config.CACHE_DIR, "processed_data.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return (
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            data["X_test"],
            data["test_ids"],
        )

    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META)
    val_meta = pd.read_csv(Config.VAL_META)
    test_meta = pd.read_csv(Config.TEST_META)

    # Load JSONs
    with open(Config.TRAIN_JSON, "r") as f:
        train_json = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_json = json.load(f)

    # Create lookup dicts for faster access
    train_dict = {item["id"]: item for item in train_json}
    test_dict = {item["id"]: item for item in test_json}

    def prepare_subset(meta_df, source_dict, is_test=False):
        X_list = []
        y_list = []
        ids_list = []

        for _, row in meta_df.iterrows():
            img_id = row["id"]
            item = source_dict[img_id]

            # Process Image
            img_vec = process_image_bands(
                item["band_1"], item["band_2"], Config.IMAGE_SIZE
            )

            # Process Inc Angle
            # Use value from metadata (already cleaned/numeric)
            inc_angle = row["inc_angle"]

            # Combine: Append inc_angle to the end of the image vector
            feat_vec = np.append(img_vec, inc_angle)

            X_list.append(feat_vec)
            ids_list.append(img_id)

            if not is_test:
                y_list.append(row["is_iceberg"])

        return (
            np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            np.array(ids_list),
        )

    # Prepare splits
    print("Preparing Train split...")
    X_train_raw, y_train, _ = prepare_subset(train_meta, train_dict)
    print("Preparing Validation split...")
    X_val_raw, y_val, _ = prepare_subset(val_meta, train_dict)
    print("Preparing Test split...")
    X_test_raw, _, test_ids = prepare_subset(test_meta, test_dict, is_test=True)

    # Impute missing inc_angle
    # The inc_angle is the last column
    inc_angle_col_idx = X_train_raw.shape[1] - 1

    train_angles = X_train_raw[:, inc_angle_col_idx]
    # Calculate median from training set ignoring NaNs
    median_angle = np.nanmedian(train_angles)

    def fill_na_angle(X, value):
        col = X[:, inc_angle_col_idx]
        mask = np.isnan(col)
        X[mask, inc_angle_col_idx] = value
        return X

    X_train_filled = fill_na_angle(X_train_raw, median_angle)
    X_val_filled = fill_na_angle(X_val_raw, median_angle)
    X_test_filled = fill_na_angle(X_test_raw, median_angle)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled)
    X_val_scaled = scaler.transform(X_val_filled)
    X_test_scaled = scaler.transform(X_test_filled)

    # Cache results
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.savez(
        cache_file,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        X_test=X_test_scaled,
        test_ids=test_ids,
    )

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, test_ids


def train_model(model, train_loader, val_loader, epochs, patience, lr):
    """
    Trains the model with Early Stopping.
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.10f} - Val Loss: {val_loss:.10f}"
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            # Save checkpoint
            torch.save(best_model_state, Config.MODEL_CHECKPOINT)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


def generate_submission(model, test_loader, test_ids):
    """
    Generates predictions and saves submission CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch[0].to(device)  # TensorDataset returns tuple
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy().flatten().tolist())

    submission = pd.DataFrame({"id": test_ids, "is_iceberg": predictions})

    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)


def run_pipeline():
    """
    Orchestrates the full pipeline: Data Loading -> Training -> Submission.
    """
    # 1. Get Data
    X_train, y_train, X_val, y_val, X_test, test_ids = get_data()

    # 2. Create DataLoaders
    batch_size = Config.BATCH_SIZE

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Initialize Model
    input_dim = Config.INPUT_DIM
    model = D2N(
        input_dim=input_dim,
        hidden_units=Config.HIDDEN_UNITS,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # 4. Train
    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
        lr=Config.LEARNING_RATE,
    )

    # 5. Predict and Submit
    generate_submission(model, test_loader, test_ids)
