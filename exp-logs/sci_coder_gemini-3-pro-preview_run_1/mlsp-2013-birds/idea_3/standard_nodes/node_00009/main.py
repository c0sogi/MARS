import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.dataset import load_data, BirdDataset, InferenceDataset
from library.model import BirdResNet
from library.engine import train_model

# Attempt to import IterativeStratification for better multi-label splits
try:
    from skmultilearn.model_selection import IterativeStratification

    HAS_SKMULTILEARN = True
except ImportError:
    HAS_SKMULTILEARN = False


def get_predictions(models, dataset, device):
    """
    Generates ensemble predictions using sliding window inference.
    Averages probabilities across all models in the list.
    """
    # Set all models to eval mode
    for model in models:
        model.eval()

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_probs = []
    all_ids = []

    crop_width = Config.CROP_WIDTH
    stride = Config.STRIDE

    with torch.no_grad():
        for inputs, _, rec_ids in loader:
            inputs = inputs.to(device)
            B, C, H, W = inputs.shape

            # Sliding Window Logic
            starts = list(range(0, W - crop_width + 1, stride))
            if W > crop_width and (starts[-1] + crop_width < W):
                starts.append(W - crop_width)
            elif W <= crop_width:
                starts = [0]

            # Create crops for the batch
            batch_crops = []
            # Map each crop back to its image index in the batch
            crop_indices = []

            for i in range(B):
                img = inputs[i]
                for s in starts:
                    crop = img[:, :, s : s + crop_width]
                    batch_crops.append(crop)
                    crop_indices.append(i)

            if len(batch_crops) > 0:
                batch_crops_tensor = torch.stack(
                    batch_crops
                )  # (Total_Crops, 3, H, W_crop)

                # Get predictions from each model
                model_probs = []
                for model in models:
                    logits = model(batch_crops_tensor)
                    probs = torch.sigmoid(logits)
                    model_probs.append(probs)

                # Average probabilities across models
                avg_crop_probs = torch.mean(
                    torch.stack(model_probs), dim=0
                )  # (Total_Crops, Num_Classes)

                # Aggregate crops back to images
                # We need to average the probabilities of crops belonging to the same image
                batch_final_probs = torch.zeros((B, Config.NUM_CLASSES), device=device)
                crop_counts = torch.zeros((B, 1), device=device)

                for c_idx, img_idx in enumerate(crop_indices):
                    batch_final_probs[img_idx] += avg_crop_probs[c_idx]
                    crop_counts[img_idx] += 1

                # Avoid division by zero (though counts should be >= 1)
                crop_counts[crop_counts == 0] = 1
                batch_final_probs /= crop_counts

            else:
                batch_final_probs = torch.zeros((B, Config.NUM_CLASSES), device=device)

            all_probs.append(batch_final_probs.cpu().numpy())
            all_ids.append(rec_ids.numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_ids, axis=0)


def main():
    # 1. Initialization
    Config.initialize()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Loading Data...")
    # Load Training Data (Pool for CV)
    X_train_full, y_train_full, ids_train_full = load_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data=True
    )

    # Load Hold-out Validation Data
    X_holdout, y_holdout, ids_holdout = load_data(
        Config.VAL_METADATA_PATH, "val", load_cached_data=True
    )

    print(f"Training Samples: {len(X_train_full)}")
    print(f"Hold-out Validation Samples: {len(X_holdout)}")

    # 2. Stratified K-Fold Setup
    n_folds = Config.N_FOLDS
    if HAS_SKMULTILEARN:
        print(f"Using IterativeStratification with {n_folds} folds.")
        # IterativeStratification requires X to be (N, D), we use dummy X
        dummy_X = np.zeros((len(y_train_full), 1))
        kfold = IterativeStratification(n_splits=n_folds, order=1)
        splits = list(kfold.split(dummy_X, y_train_full))
    else:
        print(f"Using KFold with {n_folds} folds (skmultilearn not found).")
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)
        splits = list(kfold.split(X_train_full, y_train_full))

    trained_models = []

    # 3. Training Loop
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n=== Training Fold {fold + 1}/{n_folds} ===")

        # Split Data
        X_train_fold = X_train_full[train_idx]
        y_train_fold = y_train_full[train_idx]

        X_val_fold = X_train_full[val_idx]
        y_val_fold = y_train_full[val_idx]
        ids_val_fold = ids_train_full[val_idx]

        # Create Datasets
        # Train: Random Crops + Augmentation
        train_dataset = BirdDataset(X_train_fold, y_train_fold)
        # Inner Val: Sliding Window (for Early Stopping)
        val_dataset = InferenceDataset(X_val_fold, ids_val_fold, y_val_fold)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = BirdResNet(pretrained=Config.PRETRAINED)
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.SCHEDULER_T_MAX
        )

        # Train
        best_model = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            num_epochs=Config.EPOCHS,
            patience=Config.PATIENCE,
        )

        trained_models.append(best_model)

    # 4. Ensemble Validation on Hold-out Set
    print("\n=== Running Ensemble Validation on Hold-out Set ===")
    holdout_dataset = InferenceDataset(X_holdout, ids_holdout, y_holdout)

    # Get ensemble predictions
    val_probs, val_ids = get_predictions(trained_models, holdout_dataset, device)

    # Compute Metric
    # Ensure y_holdout matches the order of val_ids if shuffling happened (it didn't, but good practice)
    # Here we just use y_holdout directly as DataLoader(shuffle=False) preserves order
    try:
        final_auc = roc_auc_score(y_holdout, val_probs, average="macro")
    except ValueError:
        final_auc = 0.5

    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Mean Absolute Error per sample
    # MAE = mean(|y_true - y_pred|) across classes
    mae_per_sample = np.mean(np.abs(y_holdout - val_probs), axis=1)

    # Calculate Image Features
    # Mean Intensity and Std Dev for each image in holdout
    img_means = []
    img_stds = []

    for img in X_holdout:
        # img is (H, W) uint8
        img_norm = img.astype(float) / 255.0
        img_means.append(np.mean(img_norm))
        img_stds.append(np.std(img_norm))

    img_means = np.array(img_means)
    img_stds = np.array(img_stds)

    # Correlations
    corr_mean = np.corrcoef(mae_per_sample, img_means)[0, 1]
    corr_std = np.corrcoef(mae_per_sample, img_stds)[0, 1]

    print(f"Correlation (Error vs Signal Mean): {corr_mean:.4f}")
    print(f"Correlation (Error vs Signal Std): {corr_std:.4f}")

    # 6. Submission
    THRESHOLD = 0.9255537489325414
    if final_auc > THRESHOLD:
        print(
            f"\nValidation Metric ({final_auc}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Load Test Data
        X_test, y_test_dummy, ids_test = load_data(
            Config.TEST_METADATA_PATH, "test", load_cached_data=True
        )

        test_dataset = InferenceDataset(X_test, ids_test, labels=None)

        # Get Predictions
        test_probs, test_rec_ids = get_predictions(trained_models, test_dataset, device)

        # Format Submission
        results = []
        for i in range(len(test_rec_ids)):
            rid = test_rec_ids[i]
            probs = test_probs[i]
            for species_idx in range(Config.NUM_CLASSES):
                sub_id = int(rid * 100 + species_idx)
                results.append({"Id": sub_id, "Probability": probs[species_idx]})

        df_sub = pd.DataFrame(results)
        df_sub = df_sub.sort_values("Id")

        df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"\nValidation Metric ({final_auc}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
