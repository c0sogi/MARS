import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from library.config import Config


# Set seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()

        # Input dimension: 75 * 75 * 2 = 11250
        self.input_dim = Config.INPUT_DIM

        # Block 1: Input -> Hidden 1
        self.block1 = nn.Sequential(
            nn.Linear(self.input_dim, Config.HIDDEN_UNITS[0]),
            nn.BatchNorm1d(Config.HIDDEN_UNITS[0]),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # Block 2: Hidden 1 -> Hidden 2
        self.block2 = nn.Sequential(
            nn.Linear(Config.HIDDEN_UNITS[0], Config.HIDDEN_UNITS[1]),
            nn.BatchNorm1d(Config.HIDDEN_UNITS[1]),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # Output Layer: Hidden 2 + Angle -> 1
        # Concatenating 1 scalar (inc_angle) to the output of block2
        self.out = nn.Sequential(nn.Linear(Config.HIDDEN_UNITS[1] + 1, 1), nn.Sigmoid())

    def forward(self, x_img, x_angle):
        # Flatten image: (Batch, 75, 75, 2) -> (Batch, 11250)
        x_img = x_img.view(x_img.size(0), -1)

        out = self.block1(x_img)
        out = self.block2(out)

        # Concatenate incidence angle
        # Ensure angle is (Batch, 1)
        x_angle = x_angle.view(x_angle.size(0), 1)
        combined = torch.cat([out, x_angle], dim=1)

        return self.out(combined)


def load_and_process_data(load_cached_data=True):
    """
    Loads raw data, processes it (flatten, impute, scale), and returns tensors.
    Implements caching using .npy files in Config.CACHE_DIR.
    """
    cache_files = {
        "X_train": os.path.join(Config.CACHE_DIR, "X_train.npy"),
        "angle_train": os.path.join(Config.CACHE_DIR, "angle_train.npy"),
        "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
        "X_val": os.path.join(Config.CACHE_DIR, "X_val.npy"),
        "angle_val": os.path.join(Config.CACHE_DIR, "angle_val.npy"),
        "y_val": os.path.join(Config.CACHE_DIR, "y_val.npy"),
        "X_test": os.path.join(Config.CACHE_DIR, "X_test.npy"),
        "angle_test": os.path.join(Config.CACHE_DIR, "angle_test.npy"),
        "test_ids": os.path.join(Config.CACHE_DIR, "test_ids.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        data = {}
        for k, v in cache_files.items():
            data[k] = np.load(v, allow_pickle=True)
        return data

    print("Processing data from scratch...")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META)
    val_meta = pd.read_csv(Config.VAL_META)
    test_meta = pd.read_csv(Config.TEST_META)

    # Load Raw JSON
    print("Loading train.json...")
    df_train_raw = pd.read_json(Config.TRAIN_JSON)
    df_train_raw.set_index("id", inplace=True)

    print("Loading test.json...")
    df_test_raw = pd.read_json(Config.TEST_JSON)
    df_test_raw.set_index("id", inplace=True)

    def process_subset(meta_df, raw_df, is_train=True):
        ids = meta_df["id"].values
        # Extract bands
        subset_raw = raw_df.loc[ids]

        b1 = np.array(subset_raw["band_1"].tolist())
        b2 = np.array(subset_raw["band_2"].tolist())

        # Stack and Flatten: (N, 75*75) + (N, 75*75) -> (N, 11250)
        # Concatenate band 1 and band 2
        X = np.concatenate([b1, b2], axis=1)

        # Angles
        # Meta df has inc_angle with NaNs where appropriate
        angles = meta_df["inc_angle"].values

        y = None
        if is_train:
            y = meta_df["is_iceberg"].values

        return X, angles, y, ids

    X_train, angle_train, y_train, _ = process_subset(train_meta, df_train_raw, True)
    X_val, angle_val, y_val, _ = process_subset(val_meta, df_train_raw, True)
    X_test, angle_test, _, test_ids = process_subset(test_meta, df_test_raw, False)

    # Impute Angles
    # Fit imputer on training angles only
    imputer = SimpleImputer(strategy="median")
    angle_train = imputer.fit_transform(angle_train.reshape(-1, 1)).flatten()
    angle_val = imputer.transform(angle_val.reshape(-1, 1)).flatten()
    angle_test = imputer.transform(angle_test.reshape(-1, 1)).flatten()

    # Scale Images
    # Fit scaler on training images only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save to cache
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["angle_train"], angle_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["angle_val"], angle_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angle_test"], angle_test)
    np.save(cache_files["test_ids"], test_ids)

    return {
        "X_train": X_train,
        "angle_train": angle_train,
        "y_train": y_train,
        "X_val": X_val,
        "angle_val": angle_val,
        "y_val": y_val,
        "X_test": X_test,
        "angle_test": angle_test,
        "test_ids": test_ids,
    }


def train_model(data_dict):
    device = torch.device(Config.DEVICE)

    # Prepare Tensors
    X_train = torch.FloatTensor(data_dict["X_train"])
    angle_train = torch.FloatTensor(data_dict["angle_train"])
    y_train = torch.FloatTensor(data_dict["y_train"]).unsqueeze(1)

    X_val = torch.FloatTensor(data_dict["X_val"])
    angle_val = torch.FloatTensor(data_dict["angle_val"])
    y_val = torch.FloatTensor(data_dict["y_val"]).unsqueeze(1)

    # Debugging subsample
    if Config.DEBUG_SAMPLE_SIZE:
        limit = Config.DEBUG_SAMPLE_SIZE
        X_train, angle_train, y_train = (
            X_train[:limit],
            angle_train[:limit],
            y_train[:limit],
        )
        X_val, angle_val, y_val = X_val[:limit], angle_val[:limit], y_val[:limit]

    # DataLoader
    train_dataset = TensorDataset(X_train, angle_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    val_dataset = TensorDataset(X_val, angle_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Model
    model = SimpleFCN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCELoss()

    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch_x, batch_angle, batch_y in train_loader:
            batch_x, batch_angle, batch_y = (
                batch_x.to(device),
                batch_angle.to(device),
                batch_y.to(device),
            )

            optimizer.zero_grad()
            outputs = model(batch_x, batch_angle)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_angle, batch_y in val_loader:
                batch_x, batch_angle, batch_y = (
                    batch_x.to(device),
                    batch_angle.to(device),
                    batch_y.to(device),
                )
                outputs = model(batch_x, batch_angle)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_dataset)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        )

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


def generate_submission(model, data_dict):
    device = torch.device(Config.DEVICE)
    model.eval()

    X_test = torch.FloatTensor(data_dict["X_test"])
    angle_test = torch.FloatTensor(data_dict["angle_test"])
    test_ids = data_dict["test_ids"]

    test_dataset = TensorDataset(X_test, angle_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    predictions = []

    with torch.no_grad():
        for batch_x, batch_angle in test_loader:
            batch_x, batch_angle = batch_x.to(device), batch_angle.to(device)
            outputs = model(batch_x, batch_angle)
            predictions.extend(outputs.cpu().numpy().flatten())

    # Create submission DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": predictions})

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
