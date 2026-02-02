import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import set_seed, calculate_roc_auc, write_submission_csv
from library.data import get_dataloaders
from library.model import create_model
from library.pipeline import (
    train_teachers,
    generate_pseudo_labels,
    train_student,
    sanitize_state_dict,
)


def main():
    # 1. Configuration and Setup
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Initialize Config
    # We use debug=False to run the full 50 epochs.
    # Since the dataset is small (208 samples), this is still extremely fast (minutes).
    config = Config(debug=False)

    # Ensure reproducibility
    set_seed(config.SEED)

    print("Starting Deep-Stem ResNet-34 Ensemble Distillation Pipeline...")

    # 2. Pipeline Execution

    # Stage 1: Train Teachers
    # Trains 3 independent models on the labeled training set.
    print("--- Stage 1: Training Teachers ---")
    teacher_paths = train_teachers(config)

    # Stage 2: Generate Pseudo Labels
    # Generates soft labels for the test set using the teacher ensemble.
    print("--- Stage 2: Generating Pseudo Labels ---")
    pseudo_df = generate_pseudo_labels(teacher_paths, config, load_cached_data=True)

    # Stage 3: Train Student
    # Trains the final student model on the combined dataset (Train + Pseudo-Test).
    print("--- Stage 3: Training Student ---")
    student_path = train_student(pseudo_df, config)

    # 3. Validation
    print("--- Validating Student Model ---")
    device = config.DEVICE

    # Load Validation Data
    dataloaders = get_dataloaders(config)
    val_loader = dataloaders["val"]

    # Load Student Model
    # We load the SWA model which is robust and generalized.
    model = create_model(config)
    if not os.path.exists(student_path):
        print(f"Error: Student model not found at {student_path}")
        return

    state_dict = torch.load(student_path, map_location=device)
    state_dict = sanitize_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_image_means = []

    # Inference on Validation Set
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Compute image mean intensity for failure analysis
            # images shape: (B, 3, H, W). Compute mean over C, H, W for each sample in batch.
            batch_means = images.mean(dim=[1, 2, 3]).cpu().numpy()
            all_image_means.append(batch_means)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_image_means = np.concatenate(all_image_means, axis=0)

    # Compute Metric
    val_auc = calculate_roc_auc(all_targets, all_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    print("--- Performing Failure Analysis ---")

    # Calculate error per sample (Mean Absolute Error across 19 classes)
    sample_errors = np.mean(np.abs(all_targets - all_preds), axis=1)

    # Feature 1: Label Cardinality (Number of active species per recording)
    label_counts = np.sum(all_targets, axis=1)

    # Feature 2: Signal Intensity (Mean pixel value of normalized spectrogram)
    # all_image_means is already computed

    # Compute Correlations
    if np.std(sample_errors) > 1e-9 and np.std(label_counts) > 1e-9:
        corr_count, _ = pearsonr(sample_errors, label_counts)
    else:
        corr_count = 0.0

    if np.std(sample_errors) > 1e-9 and np.std(all_image_means) > 1e-9:
        corr_intensity, _ = pearsonr(sample_errors, all_image_means)
    else:
        corr_intensity = 0.0

    print(f"Correlation (Error vs Label Count): {corr_count:.4f}")
    print(f"Correlation (Error vs Signal Intensity): {corr_intensity:.4f}")

    # Simple interpretation
    if abs(corr_count) > 0.2:
        direction = "Higher" if corr_count > 0 else "Lower"
        print(f"Observation: {direction} error associated with higher species count.")

    # 5. Submission
    print(f"Generating submission with Validation AUC: {val_auc}")

    # Load Test Data
    test_loader = dataloaders["test"]
    test_probs = []

    # Inference on Test Set
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Single forward pass (Student model is already robust)
            logits = model(images)
            probs = torch.sigmoid(logits)
            test_probs.append(probs.cpu().numpy())

    test_probs = np.concatenate(test_probs, axis=0)

    # Get Recording IDs
    test_df = test_loader.dataset.df
    rec_ids = test_df["rec_id"].values

    # Write Submission CSV
    write_submission_csv(rec_ids, test_probs, config.SUBMISSION_PATH)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
