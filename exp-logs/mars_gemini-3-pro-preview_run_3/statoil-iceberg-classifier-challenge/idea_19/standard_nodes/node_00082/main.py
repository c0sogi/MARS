import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import from provided libraries
from library.utils import set_seed
from library.data_loader import load_data, IcebergDataset, get_transforms
from library.model import StabilizedSECNN
from library.trainer import train_one_epoch, validate


def analyze_failures(X, angles, y_true, y_pred_prob):
    """
    Analyzes failure modes by correlating error with features.
    """
    # Calculate Log Loss contribution per sample
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)

    # Element-wise log loss
    errors = -(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))

    # Extract simple features for correlation
    # X shape: (N, 75, 75, 3)
    # Band 1 mean, Band 2 mean (averaging over H and W dimensions)
    b1_mean = np.mean(X[:, :, :, 0], axis=(1, 2))
    b2_mean = np.mean(X[:, :, :, 1], axis=(1, 2))

    df_analysis = pd.DataFrame(
        {"error": errors, "inc_angle": angles, "b1_mean": b1_mean, "b2_mean": b2_mean}
    )

    print("\nFailure Analysis (Correlation with Error):")
    correlations = df_analysis.corr()["error"].sort_values(ascending=False)
    print(correlations)
    return correlations


def main():
    # 1. Setup
    SEED = 42
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 32
    EPOCHS = 30
    PATIENCE = 8

    # 2. Load Data
    print("Loading data...")
    # load_data handles caching and imputation
    data = load_data(load_cached_data=True)

    # Merge train and val for 5-Fold CV
    X_full = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    angle_full = np.concatenate([data["angle_train"], data["angle_val"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"]], axis=0)

    X_test = data["X_test"]
    angle_test = data["angle_test"]
    ids_test = data["ids_test"]

    # 3. K-Fold Cross Validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # Storage for OOF predictions (for validation metric)
    oof_preds = np.zeros(len(y_full))
    # Storage for Test predictions
    test_preds_accum = np.zeros(len(X_test))

    # Prepare Test Loader (once)
    test_dataset = IcebergDataset(X_test, angle_test, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    print(f"Starting {n_splits}-Fold Cross-Validation on {len(X_full)} samples...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\nFold {fold + 1}/{n_splits}")

        # Split Data
        X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
        angle_train_fold, angle_val_fold = angle_full[train_idx], angle_full[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]

        # Create Datasets
        train_dataset = IcebergDataset(
            X_train_fold,
            angle_train_fold,
            y_train_fold,
            transform=get_transforms("train"),
        )
        val_dataset = IcebergDataset(
            X_val_fold, angle_val_fold, y_val_fold, transform=get_transforms("test")
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        # Initialize Model
        model = StabilizedSECNN().to(device)

        # Optimizer & Loss
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss = validate(model, val_loader, criterion, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Generate OOF Predictions for this fold
        model.eval()
        fold_oof = []
        with torch.no_grad():
            for imgs, angles, _ in val_loader:
                imgs = imgs.to(device)
                angles = angles.to(device)
                outputs = model(imgs, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_oof.extend(probs)

        oof_preds[val_idx] = np.array(fold_oof)

        # Generate Test Predictions for this fold
        fold_test_preds = []
        with torch.no_grad():
            for imgs, angles in test_loader:
                imgs = imgs.to(device)
                angles = angles.to(device)
                outputs = model(imgs, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_test_preds.extend(probs)

        test_preds_accum += np.array(fold_test_preds)

    # 4. Final Validation Metric
    final_metric = log_loss(y_full, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    analyze_failures(X_full, angle_full, y_full, oof_preds)

    # 6. Submission
    THRESHOLD = 0.18145903282502943
    if final_metric < THRESHOLD:
        print(
            f"Validation metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        avg_preds = test_preds_accum / n_splits

        os.makedirs("./submission", exist_ok=True)
        submission = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})
        submission.to_csv("./submission/submission.csv", index=False)
        print("Submission saved to ./submission/submission.csv")
    else:
        print(
            f"Validation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
