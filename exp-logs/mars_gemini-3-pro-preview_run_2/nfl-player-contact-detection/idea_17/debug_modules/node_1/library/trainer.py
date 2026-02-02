import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.features import get_data
from library.model import MMWIN, train_model, optimize_threshold, predict


class ContactDataset(Dataset):
    """
    PyTorch Dataset for the Contact Detection task.
    """

    def __init__(self, features, targets=None):
        self.features = torch.from_numpy(features).float()
        self.targets = (
            torch.from_numpy(targets).float() if targets is not None else None
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]


def get_scaled_data(load_cached_data=True):
    """
    Retrieves data and applies StandardScaler with caching mechanism.
    Strictly follows the logic: if cached files exist and load_cached_data is True, load them.
    Otherwise, compute (scale) and save.
    """
    # Define cache paths for scaled features
    cache_train = os.path.join(Config.CACHE_DIR, "X_train_scaled.npy")
    cache_val = os.path.join(Config.CACHE_DIR, "X_val_scaled.npy")
    cache_test = os.path.join(Config.CACHE_DIR, "X_test_scaled.npy")

    # Check if cached files exist
    files_exist = (
        os.path.exists(cache_train)
        and os.path.exists(cache_val)
        and os.path.exists(cache_test)
    )

    # Load raw data (needed for targets/ids anyway)
    # We pass load_cached_data to get_data as well to leverage its internal cache for raw features
    print("Retrieving feature data...")
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids = get_data(
        load_cached_data=load_cached_data
    )

    if load_cached_data and files_exist:
        print("Loading scaled features from cache...")
        X_train = np.load(cache_train)
        X_val = np.load(cache_val)
        X_test = np.load(cache_test)
    else:
        print("Scaling features (StandardScaler)...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Save to cache
        print(f"Saving scaled features to {Config.CACHE_DIR}...")
        np.save(cache_train, X_train)
        np.save(cache_val, X_val)
        np.save(cache_test, X_test)

    return X_train, y_train, X_val, y_val, X_test, test_ids


def run_training(load_cached_data=True):
    """
    Main pipeline execution: Data Loading -> Scaling -> Training -> Threshold Opt -> Inference.
    """
    seed_everything(Config.SEED)

    # 1. Prepare Data
    X_train, y_train, X_val, y_val, X_test, test_ids = get_scaled_data(load_cached_data)

    # Create Datasets
    train_dataset = ContactDataset(X_train, y_train)
    val_dataset = ContactDataset(X_val, y_val)
    test_dataset = ContactDataset(X_test)

    # Create Loaders
    # num_workers=4 is safe for 12 vCPUs
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    print(f"Initializing MMWIN model with input_dim={input_dim} on {device}...")
    model = MMWIN(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    )

    # 3. Train
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=Config.EPOCHS,
        lr=Config.LEARNING_RATE,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # 4. Optimize Threshold
    print("Optimizing decision threshold...")
    best_threshold = optimize_threshold(model, val_loader, device)

    # Validation Metrics (Full Precision)
    # Re-predict on val to get exact MCC for reporting
    model.eval()
    val_probs = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            val_probs.append(probs.cpu().numpy())

    val_probs = np.concatenate(val_probs)
    val_preds = (val_probs > best_threshold).astype(int)
    final_val_mcc = compute_mcc(y_val, val_preds)

    print(f"Final Validation MCC (Threshold {best_threshold}): {final_val_mcc}")

    # 5. Predict on Test
    print("Generating test predictions...")
    test_preds = predict(model, test_loader, device, threshold=best_threshold)

    # 6. Save Submission
    submission = pd.DataFrame({"contact_id": test_ids, "contact": test_preds})

    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
