import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library components
from library.config import (
    setup_system,
    SEED,
    METADATA_DIR,
    DEVICE,
    WORKING_DIR,
    SUBMISSION_DIR,
)
from library.utils import probabilistic_f1
from library.vision_backbone import get_image_encoder
from library.data_handler import generate_embeddings
from library.tabular_model import CancerClassifier


def main():
    print("=== Starting Breast Cancer Detection Pipeline Demo ===")

    # 1. System Setup
    # Initialize directories and set random seeds for reproducibility
    setup_system(seed=SEED)
    print(f"System setup complete. Device: {DEVICE}")

    # 2. Data Loading and Sampling
    # We load the metadata and sample a small subset to ensure the script runs quickly (within minutes)
    print("\n[Step 1] Loading and Sampling Metadata...")

    train_full = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_full = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_full = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Create a stratified sample for training to ensure we have both classes
    # Sampling 100 images total: 10 positive, 90 negative
    train_pos = train_full[train_full["cancer"] == 1].head(10)
    train_neg = train_full[train_full["cancer"] == 0].head(90)
    train_sample = (
        pd.concat([train_pos, train_neg])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    # Sample validation set
    val_pos = val_full[val_full["cancer"] == 1].head(5)
    val_neg = val_full[val_full["cancer"] == 0].head(45)
    val_sample = (
        pd.concat([val_pos, val_neg])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    # Sample test set
    test_sample = test_full.head(50).copy()

    print(
        f"Subset sizes -> Train: {len(train_sample)}, Val: {len(val_sample)}, Test: {len(test_sample)}"
    )

    # 3. Feature Extraction (Vision Backbone)
    # Initialize the frozen ResNet18 model
    print("\n[Step 2] Initializing Vision Backbone (ResNet18)...")
    backbone = get_image_encoder()

    # Generate embeddings for the subsets
    # This reads DICOMs, preprocesses them, and passes them through the ResNet
    print("Generating embeddings for Train set...")
    train_processed = generate_embeddings(train_sample, backbone, DEVICE)

    print("Generating embeddings for Validation set...")
    val_processed = generate_embeddings(val_sample, backbone, DEVICE)

    print("Generating embeddings for Test set...")
    test_processed = generate_embeddings(test_sample, backbone, DEVICE)

    # Validation: Check if embeddings were actually added
    # ResNet18 global average pooling outputs 512 dimensions
    expected_cols = 512
    emb_cols = [c for c in train_processed.columns if c.startswith("emb_")]
    assert (
        len(emb_cols) == expected_cols
    ), f"Expected {expected_cols} embedding columns, found {len(emb_cols)}"
    print(f"Successfully generated {expected_cols}-dimensional embeddings.")

    # 4. Tabular Model Training
    # Initialize the LightGBM wrapper
    print("\n[Step 3] Training Classifier...")
    clf = CancerClassifier()

    # Fit the model on the processed data (metadata + embeddings)
    # Note: LightGBM output is suppressed by verbose=-1 in config
    clf.fit(train_processed, val_processed)

    # Validation: Check if model artifacts are saved
    assert os.path.exists(clf.model_path), f"Model file missing at {clf.model_path}"
    assert os.path.exists(
        clf.encoder_path
    ), f"Encoder file missing at {clf.encoder_path}"
    print("Model training complete and artifacts saved.")

    # 5. Inference and Submission
    print("\n[Step 4] Running Inference on Test Set...")
    submission_df = clf.predict_and_submit(test_processed)

    # Validation: Check submission format
    required_cols = ["prediction_id", "cancer"]
    for col in required_cols:
        assert col in submission_df.columns, f"Submission missing column: {col}"

    # Check probability range
    probs = submission_df["cancer"]
    assert probs.min() >= 0.0 and probs.max() <= 1.0, "Predictions out of [0, 1] range"

    # Check output file
    sub_file = os.path.join(SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_file), "Submission CSV file not found on disk"
    print(f"Submission generated successfully with {len(submission_df)} rows.")

    # 6. Metric Utility Verification
    print("\n[Step 5] Verifying Metric Function (Probabilistic F1)...")
    # Test case: Perfect prediction
    y_true_perfect = np.array([0, 1, 0, 1])
    y_pred_perfect = np.array([0.0, 1.0, 0.0, 1.0])
    score_perfect = probabilistic_f1(y_true_perfect, y_pred_perfect)
    assert np.isclose(
        score_perfect, 1.0
    ), f"Expected pF1=1.0 for perfect preds, got {score_perfect}"

    # Test case: Random prediction
    y_true_mix = np.array([0, 1, 1, 0])
    y_pred_mix = np.array([0.2, 0.8, 0.6, 0.4])
    score_mix = probabilistic_f1(y_true_mix, y_pred_mix)
    print(f"Test pF1 Score: {score_mix:.4f}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
