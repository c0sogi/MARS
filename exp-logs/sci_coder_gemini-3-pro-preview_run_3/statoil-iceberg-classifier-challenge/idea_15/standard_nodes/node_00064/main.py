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

    # Hyperparameters for fast baseline
    BATCH_SIZE = 32
    EPOCHS = 30
    PATIENCE = 8
    N_FOLDS = 5
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Cite solution_lesson_node_00036: Enforce L2 Regularization

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
    # Cite solution_lesson_node_00044: Use Hold-Out Ensemble Strategy
    # Load separate train and hold-out (val) sets
    X_train, a_train, y_train, ids_train = load_data("train", load_cached_data=True)
    X_holdout, a_holdout, y_holdout, ids_holdout = load_data(
        "val", load_cached_data=True
    )

    print(f"Training samples: {len(y_train)}")
    print(f"Hold-out samples: {len(y_holdout)}")

    # -------------------------------------------------------------------------
    # 3. Cross-Validation Training (Ensemble Construction)
    # -------------------------------------------------------------------------
    # Cite solution_lesson_node_00016: Train K models using K-Fold CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    best_model_paths = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Starting Fold {fold} ---")

        # Prepare Datasets
        # Train gets augmentation, Val (internal fold val) does not
        train_ds = IcebergDataset(
            X_train[train_idx],
            a_train[train_idx],
            y_train[train_idx],
            transform=get_transforms("train"),
        )
        val_ds = IcebergDataset(
            X_train[val_idx],
            a_train[val_idx],
            y_train[val_idx],
            transform=get_transforms("val"),
        )

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )

        # Initialize Model (SimpleCNN)
        model = SimpleCNN().to(DEVICE)

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

    # -------------------------------------------------------------------------
    # 4. Validation Assessment (Hold-Out Ensemble) & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Ensemble on Hold-Out Set ---")
    # Prepare Hold-out Loader
    holdout_ds = IcebergDataset(
        X_holdout, a_holdout, y_holdout, transform=get_transforms("val")
    )
    holdout_loader = DataLoader(
        holdout_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Accumulate predictions from all fold models
    holdout_preds_accum = np.zeros((len(y_holdout), N_FOLDS))

    for i, model_path in enumerate(best_model_paths):
        model = SimpleCNN().to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for images, angles, _ in holdout_loader:
                images = images.to(DEVICE)
                angles = angles.to(DEVICE)

                outputs = model(images, angles)
                probs = torch.sigmoid(outputs)
                fold_preds.append(probs.cpu().numpy())

        holdout_preds_accum[:, i] = np.concatenate(fold_preds).flatten()

    # Average predictions (Ensemble)
    val_preds = np.mean(holdout_preds_accum, axis=1)

    final_metric = log_loss(y_holdout, val_preds)
    print(f"\nFinal Validation Metric (Hold-Out Ensemble): {final_metric}")

    print("\n--- Failure Analysis ---")
    # Calculate absolute error on hold-out set
    errors = np.abs(y_holdout - val_preds)

    # Compute stats for correlation analysis
    b1_mean = np.mean(X_holdout[:, :, :, 0], axis=(1, 2))
    b1_std = np.std(X_holdout[:, :, :, 0], axis=(1, 2))
    b2_mean = np.mean(X_holdout[:, :, :, 1], axis=(1, 2))
    b2_std = np.std(X_holdout[:, :, :, 1], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": a_holdout,
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
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
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
