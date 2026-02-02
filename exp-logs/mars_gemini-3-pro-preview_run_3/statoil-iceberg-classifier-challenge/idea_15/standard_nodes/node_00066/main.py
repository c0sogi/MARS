import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import provided library functions
from library.dataset import load_data, IcebergDataset, get_transforms, set_seed
from library.model import SimpleCNN
from library.trainer import train_fold


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------------------------
    set_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 35
    PATIENCE = 10
    N_FOLDS = 5
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Directories
    CHECKPOINT_DIR = "./working/idea_16/checkpoints"
    SUBMISSION_DIR = "./submission"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print(f"Running on device: {DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading training and validation data...")
    # Load train and val splits using provided library
    # We use X_train for CV and X_val as a fixed Hold-Out set (Cite 00044)
    X_train, a_train, y_train, ids_train = load_data("train", load_cached_data=True)
    X_val, a_val, y_val, ids_val = load_data("val", load_cached_data=True)

    print(f"Train samples: {len(y_train)}")
    print(f"Hold-out Val samples: {len(y_val)}")

    # -------------------------------------------------------------------------
    # 3. Cross-Validation Training (on X_train only)
    # -------------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    best_model_paths = []

    # Store ensemble predictions on the hold-out set
    val_preds_accum = np.zeros((len(y_val), N_FOLDS))

    for fold, (train_idx, inner_val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Starting Fold {fold} ---")

        # Prepare Datasets
        train_ds = IcebergDataset(
            X_train[train_idx],
            a_train[train_idx],
            y_train[train_idx],
            transform=get_transforms("train"),
        )
        inner_val_ds = IcebergDataset(
            X_train[inner_val_idx],
            a_train[inner_val_idx],
            y_train[inner_val_idx],
            transform=get_transforms("val"),
        )

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        inner_val_loader = DataLoader(
            inner_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        # Initialize Model (SimpleCNN - Cite 00031)
        model = SimpleCNN().to(DEVICE)

        # Train Fold
        best_loss, best_path = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=inner_val_loader,
            device=DEVICE,
            fold_idx=fold,
            epochs=EPOCHS,
            patience=PATIENCE,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            checkpoint_dir=CHECKPOINT_DIR,
        )
        best_model_paths.append(best_path)

        # Predict on Hold-Out Set (X_val) for this fold
        model.load_state_dict(torch.load(best_path))
        model.eval()

        # Create loader for hold-out set
        val_ds = IcebergDataset(X_val, a_val, y_val, transform=get_transforms("val"))
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        fold_probs = []
        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(DEVICE)
                angles = angles.to(DEVICE)
                outputs = model(images, angles)
                probs = torch.sigmoid(outputs)
                fold_probs.append(probs.cpu().numpy())

        val_preds_accum[:, fold] = np.concatenate(fold_probs).flatten()

    # -------------------------------------------------------------------------
    # 4. Validation Assessment & Failure Analysis
    # -------------------------------------------------------------------------
    # Ensemble averaging (Cite 00027)
    ensemble_val_preds = np.mean(val_preds_accum, axis=1)
    final_metric = log_loss(y_val, ensemble_val_preds)
    print(f"\nFinal Validation Metric (Hold-Out Ensemble): {final_metric}")

    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(y_val - ensemble_val_preds)

    # Compute stats for correlation analysis
    b1_mean = np.mean(X_val[:, :, :, 0], axis=(1, 2))
    b1_std = np.std(X_val[:, :, :, 0], axis=(1, 2))
    b2_mean = np.mean(X_val[:, :, :, 1], axis=(1, 2))
    b2_std = np.std(X_val[:, :, :, 1], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": a_val,
            "b1_mean": b1_mean,
            "b1_std": b1_std,
            "b2_mean": b2_mean,
            "b2_std": b2_std,
        }
    )

    # Calculate correlation with error
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    TARGET_THRESHOLD = 0.18145903282502943

    if final_metric < TARGET_THRESHOLD:
        print(
            f"\nMetric check passed ({final_metric} < {TARGET_THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        X_test, a_test, _, ids_test = load_data("test", load_cached_data=True)

        # Initialize Dataset without labels (labels=None)
        test_ds = IcebergDataset(
            X_test, a_test, labels=None, transform=get_transforms("test")
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        # Ensemble Inference
        test_preds_accum = np.zeros((len(ids_test), N_FOLDS))

        for i, model_path in enumerate(best_model_paths):
            print(f"Predicting with model from Fold {i}...")
            model = SimpleCNN().to(DEVICE)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for images, angles in test_loader:
                    images = images.to(DEVICE)
                    angles = angles.to(DEVICE)

                    outputs = model(images, angles)
                    probs = torch.sigmoid(outputs)
                    fold_preds.append(probs.cpu().numpy())

            test_preds_accum[:, i] = np.concatenate(fold_preds).flatten()

        # Average predictions
        avg_preds = np.mean(test_preds_accum, axis=1)

        # Save Submission
        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})
        df_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric check failed ({final_metric} >= {TARGET_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
