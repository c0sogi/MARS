import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import (
    SEED,
    DEVICE,
    NUM_FOLDS,
    EPOCHS,
    PATIENCE,
    BATCH_SIZE,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    MODEL_SAVE_DIR,
    SUBMISSION_PATH,
)
from library.utils import seed_everything
from library.data import load_expert_data, SDCDataset, get_transforms
from library.model import EfficientNetExpert
from library.engine import train_expert, predict_expert


def main():
    # 1. Setup and Reproducibility
    seed_everything(SEED)
    print(f"Starting Spatially-Decomposed Consensus Pipeline on {DEVICE}...")

    # We focus on the single best heuristic (center) to avoid redundancy (Cite Lesson 00018)
    experts = ["center"]
    model_records = []

    # 2. Training Loop: 3 Experts x 5 Folds
    # We use the training metadata to perform cross-validation
    print("\n" + "=" * 40)
    print(" TRAINING PHASE ")
    print("=" * 40)

    for expert in experts:
        print(f"\n>>> Processing Expert: {expert.upper()}")

        # Load training data for the specific spatial expert
        # This handles caching automatically
        images, labels, ids = load_expert_data(
            expert, "train", TRAIN_METADATA_PATH, load_cached_data=True
        )

        # Initialize Stratified K-Fold
        skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

        for fold_idx, (train_index, val_index) in enumerate(skf.split(images, labels)):
            print(f"\nTraining Fold {fold_idx + 1}/{NUM_FOLDS} for {expert}...")

            # Split data
            X_train, X_val = images[train_index], images[val_index]
            y_train, y_val = labels[train_index], labels[val_index]

            # Create Datasets
            train_dataset = SDCDataset(
                X_train, y_train, transform=get_transforms("train")
            )
            val_dataset = SDCDataset(X_val, y_val, transform=get_transforms("val"))

            # Create DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            # Train the expert model
            # The engine handles the training loop, validation, and saving the best model
            best_auc = train_expert(
                expert,
                fold_idx,
                train_loader,
                val_loader,
                epochs=EPOCHS,
                patience=PATIENCE,
            )

            # Record the path of the saved model for later inference
            model_path = os.path.join(
                MODEL_SAVE_DIR, f"best_model_{expert}_fold{fold_idx}.pth"
            )
            model_records.append(
                {"expert": expert, "fold": fold_idx, "path": model_path}
            )

    # 3. Validation Phase: Ensemble Inference on Hold-out Set
    print("\n" + "=" * 40)
    print(" VALIDATION PHASE ")
    print("=" * 40)

    # We need the ground truth labels from the validation set
    # Loading 'center' expert data to get labels and IDs (labels are same across experts)
    _, val_targets_ref, val_ids_ref = load_expert_data(
        "center", "val", VAL_METADATA_PATH, load_cached_data=True
    )

    # Initialize ensemble accumulator
    ensemble_preds = np.zeros(len(val_targets_ref))
    model_count = 0

    print(f"Aggregating predictions from {len(model_records)} models...")

    for record in model_records:
        expert = record["expert"]
        model_path = record["path"]

        # Load validation images for this specific expert
        val_images, _, _ = load_expert_data(
            expert, "val", VAL_METADATA_PATH, load_cached_data=True
        )

        # Initialize Model
        model = EfficientNetExpert(num_classes=1).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        # Create DataLoader
        dataset = SDCDataset(val_images, labels=None, transform=get_transforms("val"))
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Predict
        preds = predict_expert(model, loader, DEVICE)

        # Accumulate
        ensemble_preds += preds
        model_count += 1

    # Compute Average
    final_val_probs = ensemble_preds / model_count

    # Compute Metric
    final_auc = roc_auc_score(val_targets_ref, final_val_probs)
    print(f"Final Validation Metric: {final_auc}")

    # 4. Failure Analysis
    print("\n" + "=" * 40)
    print(" FAILURE ANALYSIS ")
    print("=" * 40)

    # Calculate absolute errors
    errors = np.abs(final_val_probs - val_targets_ref)

    # Feature 1: Mean Intensity of Center Slice
    # We load the center images again to compute simple statistics
    center_images, _, _ = load_expert_data(
        "center", "val", VAL_METADATA_PATH, load_cached_data=True
    )
    # Compute mean intensity per image (normalized 0-1)
    mean_intensities = center_images.mean(axis=(1, 2, 3))

    # Correlation between Error and Mean Intensity
    corr_intensity = np.corrcoef(errors, mean_intensities)[0, 1]
    print(f"Correlation between Error and Input Mean Intensity: {corr_intensity}")

    # Correlation between Error and Target (Class Bias)
    corr_target = np.corrcoef(errors, val_targets_ref)[0, 1]
    print(f"Correlation between Error and Target Label: {corr_target}")

    # 5. Submission Phase
    THRESHOLD = 0.6705454545454544

    if final_auc > THRESHOLD:
        print("\n" + "=" * 40)
        print(" SUBMISSION PHASE ")
        print("=" * 40)

        # Get Test IDs
        _, _, test_ids_ref = load_expert_data(
            "center", "test", TEST_METADATA_PATH, load_cached_data=True
        )

        test_ensemble_preds = np.zeros(len(test_ids_ref))

        print("Generating predictions for Test Set...")

        for record in model_records:
            expert = record["expert"]
            model_path = record["path"]

            # Load test images for this expert
            test_images, _, _ = load_expert_data(
                expert, "test", TEST_METADATA_PATH, load_cached_data=True
            )

            # Load Model
            model = EfficientNetExpert(num_classes=1).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()

            # DataLoader
            dataset = SDCDataset(
                test_images, labels=None, transform=get_transforms("test")
            )
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            # Predict
            preds = predict_expert(model, loader, DEVICE)
            test_ensemble_preds += preds

        # Average
        final_test_probs = test_ensemble_preds / model_count

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"BraTS21ID": test_ids_ref, "MGMT_value": final_test_probs}
        )

        # Save
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation AUC ({final_auc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
