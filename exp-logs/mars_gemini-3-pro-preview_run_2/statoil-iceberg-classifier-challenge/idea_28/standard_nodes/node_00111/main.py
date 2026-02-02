import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, load_data, get_device
from library.model import DIDPNet
from library.train import train_all_folds
from library.data import get_dataloaders


def run_pipeline():
    # ==========================================
    # 1. CONFIGURATION & SETUP
    # ==========================================
    # Override Config for fast baseline execution
    Config.NUM_EPOCHS = 30
    Config.PATIENCE = 8
    Config.SCHEDULER_PATIENCE = 3

    # Setup directories and seeds
    Config.setup()
    seed_everything(Config.SEED)
    device = get_device()

    print("Starting DIDPNet Pipeline...")
    print(f"Device: {device}")

    # ==========================================
    # 2. TRAINING
    # ==========================================
    # Train 5 folds. This saves models to disk.
    print("\n>>> Phase 1: Training Models")
    train_all_folds(debug=False)

    # ==========================================
    # 3. HOLD-OUT VALIDATION
    # ==========================================
    print("\n>>> Phase 2: Hold-out Validation Assessment")

    # Load Metadata for Hold-out Validation
    val_meta_path = Config.VAL_META
    if not os.path.exists(val_meta_path):
        raise FileNotFoundError(f"Validation metadata not found at {val_meta_path}")

    df_val_meta = pd.read_csv(val_meta_path)
    val_ids_set = set(df_val_meta["id"].values)

    # Load Raw Data (Cached)
    data = load_data(Config)
    all_train_images = data["train_images"]
    all_train_angles = data["train_angles"]
    all_train_labels = data["train_labels"]
    all_train_ids = data["train_ids"]

    # Identify indices for the hold-out set
    # We map ID -> Index
    id_to_idx = {id_: i for i, id_ in enumerate(all_train_ids)}
    val_indices = [
        id_to_idx[uid] for uid in df_val_meta["id"].values if uid in id_to_idx
    ]

    if len(val_indices) != len(df_val_meta):
        print(
            f"Warning: Found {len(val_indices)} samples out of {len(df_val_meta)} in metadata."
        )

    X_val_raw = all_train_images[val_indices]
    y_val_raw = all_train_labels[val_indices]
    a_val_raw = all_train_angles[val_indices]

    # Prepare to ensemble predictions
    val_preds_accum = np.zeros(len(val_indices))

    # Iterate through folds to predict on hold-out set
    # We must replicate the scaling logic used during training for each fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    folds = list(skf.split(all_train_images, all_train_labels))

    for fold_idx in range(Config.N_FOLDS):
        print(f"Validating with Fold {fold_idx} model...")

        # Reconstruct Training Split for Scaling Stats
        train_idx, _ = folds[fold_idx]
        X_train_fold = all_train_images[train_idx]
        a_train_fold = all_train_angles[train_idx]

        # Calculate Scaling Stats (Min-Max)
        min_stat = np.min(X_train_fold, axis=(0, 2, 3)).reshape(1, 3, 1, 1)
        max_stat = np.max(X_train_fold, axis=(0, 2, 3)).reshape(1, 3, 1, 1)
        range_stat = max_stat - min_stat
        range_stat[range_stat == 0] = 1.0

        # Calculate Angle Imputation Stat
        angle_mean = np.nanmean(a_train_fold)

        # Apply Preprocessing to Hold-out Set
        X_val_scaled = (X_val_raw - min_stat) / range_stat
        a_val_imputed = np.nan_to_num(a_val_raw, nan=angle_mean)

        # Create Tensor Dataset
        val_tensor_x = torch.from_numpy(X_val_scaled).float()
        val_tensor_a = torch.from_numpy(a_val_imputed).float()
        val_dataset = TensorDataset(val_tensor_x, val_tensor_a)
        val_loader = DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )

        # Load Model
        model = DIDPNet(
            backbone_filters=Config.BACKBONE_FILTERS, dropout_rate=Config.DROPOUT_RATE
        )
        model_path = Config.MODEL_PATH_TEMPLATE.format(fold_idx)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Inference
        fold_preds = []
        with torch.no_grad():
            for images, angles in val_loader:
                images = images.to(device)
                angles = angles.to(device)

                logits = model(images, angles)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                fold_preds.extend(probs)

        val_preds_accum += np.array(fold_preds)

    # Average Predictions
    val_preds_avg = val_preds_accum / Config.N_FOLDS

    # Calculate Metric
    final_log_loss = log_loss(y_val_raw, val_preds_avg)
    print(f"Final Validation Metric: {final_log_loss}")

    # ==========================================
    # 4. FAILURE ANALYSIS
    # ==========================================
    print("\n>>> Phase 3: Failure Analysis")

    # Calculate Error Magnitude
    errors = np.abs(y_val_raw - val_preds_avg)

    # Feature 1: Incidence Angle (Imputed for analysis)
    # Use global mean for NaN in analysis to avoid dropping rows
    global_angle_mean = np.nanmean(a_val_raw)
    angles_clean = np.nan_to_num(a_val_raw, nan=global_angle_mean)

    # Feature 2: Image Brightness (Mean of all channels)
    # X_val_raw is (N, 3, 75, 75)
    image_means = np.mean(X_val_raw, axis=(1, 2, 3))

    # Correlations
    corr_angle, _ = pearsonr(errors, angles_clean)
    corr_brightness, _ = pearsonr(errors, image_means)

    print("Correlation between Error Magnitude and Input Features:")
    print(f"  Incidence Angle: {corr_angle:.4f}")
    print(f"  Image Brightness: {corr_brightness:.4f}")

    # ==========================================
    # 5. SUBMISSION
    # ==========================================
    THRESHOLD = 0.16676861786296204

    if final_log_loss < THRESHOLD:
        print("\n>>> Phase 4: Generating Submission")

        # We need to predict on the test set
        # We use get_dataloaders to get the test loader for each fold
        # (which handles the correct scaling for that fold automatically)

        # Get test IDs from the first fold loader (order is consistent)
        _, _, _, test_ids = get_dataloaders(Config, fold_index=0)
        test_preds_accum = np.zeros(len(test_ids))

        for fold_idx in range(Config.N_FOLDS):
            print(f"Predicting Test Set with Fold {fold_idx} model...")

            # Get loader with correct scaling
            _, _, test_loader, _ = get_dataloaders(Config, fold_index=fold_idx)

            # Load Model
            model = DIDPNet(
                backbone_filters=Config.BACKBONE_FILTERS,
                dropout_rate=Config.DROPOUT_RATE,
            )
            model_path = Config.MODEL_PATH_TEMPLATE.format(fold_idx)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for images, angles in test_loader:
                    images = images.to(device)
                    angles = angles.to(device)

                    logits = model(images, angles)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    fold_preds.extend(probs)

            test_preds_accum += np.array(fold_preds)

        # Average
        test_preds_avg = test_preds_accum / Config.N_FOLDS

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": test_ids, "is_iceberg": test_preds_avg})

        # Save
        save_path = Config.SUBMISSION_PATH
        submission.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
        print(submission.head())

    else:
        print(
            f"\nValidation metric ({final_log_loss}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
