import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import provided library functions
from library.dataset import load_data, IcebergDataset, get_transforms, set_seed
from library.model import SHSE_CNN
from library.trainer import train_fold


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------------------------
    set_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters for fast baseline
    BATCH_SIZE = 32
    EPOCHS = 30
    PATIENCE = 8
    N_FOLDS = 5
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Directories
    CHECKPOINT_DIR = "./working/idea_15/checkpoints"
    SUBMISSION_DIR = "./submission"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print(f"Running on device: {DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading training and validation data...")
    # Load train and val splits using provided library
    X_train, a_train, y_train, ids_train = load_data("train", load_cached_data=True)
    X_val, a_val, y_val, ids_val = load_data("val", load_cached_data=True)

    # Concatenate for Stratified K-Fold
    X_all = np.concatenate([X_train, X_val], axis=0)
    a_all = np.concatenate([a_train, a_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    # ids are not strictly needed for training, but good for tracking
    ids_all = np.concatenate([ids_train, ids_val], axis=0)

    print(f"Total samples: {len(y_all)}")

    # -------------------------------------------------------------------------
    # 3. Cross-Validation Training
    # -------------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Array to store Out-Of-Fold predictions
    oof_preds = np.zeros(len(y_all))
    best_model_paths = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n--- Starting Fold {fold} ---")

        # Prepare Datasets
        # Train gets augmentation, Val does not
        train_ds = IcebergDataset(
            X_all[train_idx],
            a_all[train_idx],
            y_all[train_idx],
            transform=get_transforms("train"),
        )
        val_ds = IcebergDataset(
            X_all[val_idx],
            a_all[val_idx],
            y_all[val_idx],
            transform=get_transforms("val"),
        )

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        # Initialize Model
        model = SHSE_CNN().to(DEVICE)

        # Train Fold
        best_loss, best_path = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            fold_idx=fold,
            epochs=EPOCHS,
            patience=PATIENCE,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            checkpoint_dir=CHECKPOINT_DIR,
        )
        best_model_paths.append(best_path)

        # Generate OOF Predictions for this fold
        # Load best model state
        model.load_state_dict(torch.load(best_path))
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(DEVICE)
                angles = angles.to(DEVICE)

                outputs = model(images, angles)
                probs = torch.sigmoid(outputs)
                fold_probs.append(probs.cpu().numpy())

        oof_preds[val_idx] = np.concatenate(fold_probs).flatten()

    # -------------------------------------------------------------------------
    # 4. Validation Assessment & Failure Analysis
    # -------------------------------------------------------------------------
    final_metric = log_loss(y_all, oof_preds)
    print(f"\nFinal Validation Metric: {final_metric}")

    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(y_all - oof_preds)

    # Compute stats for correlation analysis
    # X_all is (N, 75, 75, 3). Band 0 is HH, Band 1 is HV.
    b1_mean = np.mean(X_all[:, :, :, 0], axis=(1, 2))
    b1_std = np.std(X_all[:, :, :, 0], axis=(1, 2))
    b2_mean = np.mean(X_all[:, :, :, 1], axis=(1, 2))
    b2_std = np.std(X_all[:, :, :, 1], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": a_all,
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
        # Note: load_data returns dummy labels for test, so we ignore them in dataset init
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
            model = SHSE_CNN().to(DEVICE)
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
