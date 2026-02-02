import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.stats import pointbiserialr, pearsonr

# Import from provided library files
from library.config import Config
from library.dataset import load_data, CatDogDataset, get_transforms
from library.models import build_model
from library.engine import train_model, predict_with_tta, set_seed


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Load Training Data
    # load_data("train") returns combined train + val metadata suitable for CV
    train_df = load_data("train", load_cached_data=True)
    print(f"Loaded training data: {len(train_df)} samples")

    # 3. Stratified K-Fold Setup
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Array to store OOF predictions (probabilities of class 1)
    # Initialize with -1 to detect unpredicted indices if any logic error occurs
    oof_preds = np.full(len(train_df), -1.0)

    # 4. Training Loop
    print(f"Starting {Config.N_FOLDS}-Fold Stratified Cross-Validation...")

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df["label"])
    ):
        print(f"\n--- Fold {fold_idx + 1}/{Config.N_FOLDS} ---")

        # Split Data
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        # Train: Augmentation enabled
        train_dataset = CatDogDataset(fold_train_df, transforms=get_transforms("train"))
        # Val: Deterministic resize
        val_dataset = CatDogDataset(fold_val_df, transforms=get_transforms("valid"))

        # Create Loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Store predictions from different backbones for this fold to average later
        fold_backbone_preds = []

        for backbone in Config.MODEL_BACKBONES:
            print(f"Training Backbone: {backbone}")

            # Build Model
            model = build_model(backbone, pretrained=Config.PRETRAINED)
            model = model.to(device)

            # Train (returns model with best weights loaded)
            model, best_loss = train_model(
                model, train_loader, val_loader, device, fold_idx, backbone
            )

            # Generate OOF predictions for this backbone using TTA
            # We use predict_with_tta to ensure OOF quality matches Test inference
            probs, _ = predict_with_tta(model, val_loader, device)
            fold_backbone_preds.append(probs)

            # Clean up to save memory
            del model
            torch.cuda.empty_cache()

        # Average predictions across backbones for this fold (Ensemble within fold)
        avg_fold_preds = np.mean(fold_backbone_preds, axis=0)

        # Store in global OOF array
        # val_idx comes from skf.split which indexes into the original train_df
        oof_preds[val_idx] = avg_fold_preds

    # 5. Validation Analysis
    print("\n--- Validation Analysis ---")
    y_true = train_df["label"].values

    # Calculate Log Loss (Final Validation Metric)
    final_metric = log_loss(y_true, oof_preds)
    # Print full precision as required
    print(f"Final Validation Metric: {final_metric:.16f}")

    # Failure Analysis
    print("Performing Failure Analysis...")
    errors = np.abs(y_true - oof_preds)

    # Extract simple features for correlation analysis
    file_sizes = []

    # Efficiently gather file sizes
    for idx, row in train_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["filepath"])
        try:
            file_sizes.append(os.path.getsize(full_path))
        except OSError:
            file_sizes.append(0)

    # Correlation: Error vs File Size
    corr_size, p_size = pearsonr(errors, file_sizes)
    print(f"Correlation (Error vs File Size): {corr_size:.4f} (p={p_size:.4f})")

    # Correlation: Error vs Label (Class Bias)
    corr_label, p_label = pointbiserialr(errors, y_true)
    print(f"Correlation (Error vs Label): {corr_label:.4f} (p={p_label:.4f})")

    # 6. Submission
    THRESHOLD = 0.009311713870561527

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric:.6f} < {THRESHOLD}. Generating submission...")

        # Load Test Data
        test_df = load_data("test", load_cached_data=True)
        test_dataset = CatDogDataset(test_df, transforms=get_transforms("valid"))
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Accumulate predictions from all 10 models (5 folds * 2 backbones)
        test_probs_sum = np.zeros(len(test_df))
        model_count = 0
        final_ids = None

        for fold_idx in range(Config.N_FOLDS):
            for backbone in Config.MODEL_BACKBONES:
                model_path = os.path.join(
                    Config.WORKING_DIR, f"{backbone}_fold_{fold_idx}.pth"
                )

                if not os.path.exists(model_path):
                    print(f"Warning: Model checkpoint {model_path} not found.")
                    continue

                # Load Model
                model = build_model(backbone, pretrained=False)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model = model.to(device)

                # Predict
                probs, ids = predict_with_tta(model, test_loader, device)
                test_probs_sum += probs
                model_count += 1

                # Store IDs from the first successful prediction to ensure alignment
                if final_ids is None:
                    final_ids = ids

                del model
                torch.cuda.empty_cache()

        if model_count > 0:
            avg_test_probs = test_probs_sum / model_count

            # Create Submission DataFrame
            submission_df = pd.DataFrame({"id": final_ids, "label": avg_test_probs})

            submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")
        else:
            print("Error: No models were loaded for inference.")

    else:
        print(f"\nMetric {final_metric:.6f} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
