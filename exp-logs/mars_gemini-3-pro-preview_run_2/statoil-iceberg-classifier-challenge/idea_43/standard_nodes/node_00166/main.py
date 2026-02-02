import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.model import DMWBNet
from library.data import process_data, IcebergDataset
from library.train import Trainer, EarlyStopping


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # Load cached preprocessed data
    # Returns: X_train (N, 3, 75, 75), y_train (N,), inc_train (N,), X_test, inc_test, test_ids
    X, y, inc, X_test, inc_test, test_ids = process_data(load_cached_data=True)

    # Reload raw train.json to get IDs for mapping metadata
    # This is necessary because process_data does not return train_ids
    with open(Config.TRAIN_JSON, "r") as f:
        train_data_raw = json.load(f)
    # Ensure order matches process_data (which converts to DF then extracts)
    train_ids = np.array([item["id"] for item in train_data_raw])

    # 3. 5-Fold Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Arrays to store results
    oof_preds = np.zeros(len(X))
    test_preds_accumulator = np.zeros(len(X_test))

    # Training Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Data Slicing
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        inc_tr, inc_val = inc[train_idx], inc[val_idx]

        # Dataset & Loader Creation
        train_ds = IcebergDataset(X_tr, inc_tr, y_tr, transform=True)
        val_ds = IcebergDataset(X_val, inc_val, y_val, transform=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing overhead in script
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
        )

        # Model Initialization
        model = DMWBNet().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        # Training
        early_stopping = EarlyStopping(patience=Config.PATIENCE)
        trainer = Trainer(model, device, criterion, optimizer, scheduler)

        # Fit model
        trainer.fit(train_loader, val_loader, Config.EPOCHS, early_stopping)

        # Load best weights for inference
        model.load_state_dict(early_stopping.best_model_wts)
        model.eval()

        # Inference: Validation (OOF)
        fold_oof_preds = []
        with torch.no_grad():
            for inputs, angles, _ in val_loader:
                inputs = inputs.to(device)
                angles = angles.to(device)
                outputs = model(inputs, angles)
                fold_oof_preds.extend(outputs.cpu().numpy().flatten())
        oof_preds[val_idx] = fold_oof_preds

        # Inference: Test
        test_ds = IcebergDataset(X_test, inc_test, transform=False)
        test_loader = DataLoader(
            test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
        )

        fold_test_preds = []
        with torch.no_grad():
            for inputs, angles in test_loader:
                inputs = inputs.to(device)
                angles = angles.to(device)
                outputs = model(inputs, angles)
                fold_test_preds.extend(outputs.cpu().numpy().flatten())
        test_preds_accumulator += np.array(fold_test_preds)

    # 4. Validation Assessment
    # Load metadata to identify the specific validation set
    val_meta_path = os.path.join("./metadata", "val.csv")
    if os.path.exists(val_meta_path):
        val_meta_df = pd.read_csv(val_meta_path)
        val_target_ids = val_meta_df["id"].values

        # Map IDs to indices in the full training set
        id_to_idx = {id_: i for i, id_ in enumerate(train_ids)}
        target_indices = [id_to_idx[id_] for id_ in val_target_ids if id_ in id_to_idx]

        # Extract predictions and true labels for the validation set
        val_preds_subset = oof_preds[target_indices]
        val_y_subset = y[target_indices]

        # Clip predictions to prevent log(0)
        val_preds_subset = np.clip(val_preds_subset, 1e-15, 1 - 1e-15)

        # Compute Metric
        final_metric = log_loss(val_y_subset, val_preds_subset)
        print(f"Final Validation Metric: {final_metric}")

        # 5. Failure Analysis
        print("\nFailure Analysis:")
        errors = np.abs(val_y_subset - val_preds_subset)

        # Extract features for analysis
        inc_subset = inc[target_indices]
        X_subset = X[target_indices]

        # Compute simple image stats (Mean intensity per band)
        # X is (N, 3, 75, 75). Channel 0 = Band 1, Channel 1 = Band 2
        b1_mean = X_subset[:, 0, :, :].mean(axis=(1, 2))
        b2_mean = X_subset[:, 1, :, :].mean(axis=(1, 2))

        df_fail = pd.DataFrame(
            {
                "error": errors,
                "inc_angle": inc_subset,
                "b1_mean": b1_mean,
                "b2_mean": b2_mean,
            }
        )

        # Compute correlations
        correlations = df_fail.corr()["error"].drop("error")
        print("Correlation between Error and Features:")
        print(correlations)

        # 6. Submission Generation
        THRESHOLD = 0.15417750501723176
        if final_metric < THRESHOLD:
            avg_test_preds = test_preds_accumulator / Config.N_FOLDS
            submission = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_preds})

            os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
            submission.to_csv(Config.SUBMISSION_FILE, index=False)
            print(f"Submission saved to {Config.SUBMISSION_FILE}")
        else:
            print(
                f"Validation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
            )

    else:
        print("Error: Metadata validation file not found.")


if __name__ == "__main__":
    run()
