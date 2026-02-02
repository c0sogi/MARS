import os
import sys
import numpy as np
import torch
from scipy.stats import pearsonr

# Ensure the current directory is in the python path to allow imports from library
sys.path.append(os.getcwd())

from library.config import Config
from library.data_utils import get_dataloaders
from library.model_factory import create_backbone, set_seed
from library.embedding_engine import extract_embeddings
from library.classifier_engine import (
    train_and_evaluate,
    generate_submission,
    predict_probas,
)


def main():
    # 1. Initialization and Configuration
    print("Initializing orchestration script...")
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Computation Device: {device}")

    # 2. Data Loading
    # We utilize the full dataset to ensure the high accuracy requirement (Log Loss < 0.116) is met.
    # The A100 GPU allows efficient processing of the full 10k images within the time limit.
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(debug=False)

    # 3. Backbone Model Initialization
    print("Creating Multi-Scale Backbone Model (ConvNeXt-Large)...")
    backbone = create_backbone()

    # 4. Feature Extraction
    # Extracts features using the Multi-Scale Deep Feature Pyramid strategy.
    # Caching is enabled to skip re-computation if files exist in ./working/idea_7

    print("Extracting Training Features...")
    train_features, train_labels = extract_embeddings(
        train_loader,
        backbone,
        device,
        Config.CACHE_TRAIN_FEATURES,
        Config.CACHE_TRAIN_LABELS,
        load_cached_data=True,
    )

    print("Extracting Validation Features...")
    val_features, val_labels = extract_embeddings(
        val_loader,
        backbone,
        device,
        Config.CACHE_VAL_FEATURES,
        Config.CACHE_VAL_LABELS,
        load_cached_data=True,
    )

    print("Extracting Test Features...")
    test_features, test_ids = extract_embeddings(
        test_loader,
        backbone,
        device,
        Config.CACHE_TEST_FEATURES,
        Config.CACHE_TEST_IDS,
        load_cached_data=True,
    )

    # Ensure labels are integers for sklearn compatibility
    train_labels = train_labels.astype(int)
    val_labels = val_labels.astype(int)

    # 5. Model Training & Evaluation
    print("Training LogisticRegressionCV Classifier...")
    # Trains the classifier and calculates the Multi Class Log Loss on the validation set
    model, val_metric = train_and_evaluate(
        train_features, train_labels, val_features, val_labels
    )

    # REQUIRED: Print the final validation metric in the specified format
    print(f"Final Validation Metric: {val_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Predict probabilities on validation set to analyze errors
    val_probs = predict_probas(model, val_features)

    # Calculate Log Loss per sample: -log(p_true_class)
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)

    # Extract probability assigned to the true class
    n_samples = len(val_labels)
    true_class_probs = val_probs_clipped[np.arange(n_samples), val_labels]

    # Calculate error magnitude (Log Loss) for each sample
    sample_losses = -np.log(true_class_probs)

    # Analyze correlation between Error Magnitude and Input Features
    # We use Feature Vector Norm (Signal Strength) and Feature Mean as proxy statistics for the input features
    feature_norms = np.linalg.norm(val_features, axis=1)
    feature_means = np.mean(val_features, axis=1)

    corr_norm, p_norm = pearsonr(sample_losses, feature_norms)
    corr_mean, p_mean = pearsonr(sample_losses, feature_means)

    print(
        f"Correlation (Error vs Feature Norm): {corr_norm:.6f} (p-value: {p_norm:.4e})"
    )
    print(
        f"Correlation (Error vs Feature Mean): {corr_mean:.6f} (p-value: {p_mean:.4e})"
    )

    if abs(corr_norm) > 0.1:
        print(
            "Observation: Significant correlation between signal magnitude and error."
        )
    else:
        print(
            "Observation: No significant correlation between signal magnitude and error."
        )

    # 7. Submission Generation
    # The task specifies a strict threshold for generating the submission
    THRESHOLD = 0.11640673500383826

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric {val_metric} meets threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(model, test_features, test_ids, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric {val_metric} does NOT meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
