import os
import sys
import numpy as np
import pandas as pd
import torch

# Import provided library classes
from library.config import Config
from library.data_utils import LeafImageDataset
from library.feature_extractor import DualStreamExtractor
from library.preprocessor import ManifoldProcessor
from library.classifier import LDAManager


def main():
    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    print("--- 1. Setup & Configuration ---")

    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 12  # Small subset for speed
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = "./working/demo_submission.csv"

    # Initialize directories
    Config.setup()

    # Set seeds for reproducibility
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    print("\n--- 2. Data Loading Demonstration ---")

    # Initialize Dataset (Train)
    dataset = LeafImageDataset(Config.TRAIN_META_PATH)
    print(f"Dataset initialized with {len(dataset)} samples.")

    # Fetch a single sample
    sample = dataset[0]

    # Display shapes
    print(f"Sample ID: {sample['id']}")
    print(
        f"DINO Views Shape:     {sample['dino_views'].shape}"
    )  # Expected: (4, 3, 518, 518)
    print(
        f"ConvNeXt Views Shape: {sample['convnext_views'].shape}"
    )  # Expected: (4, 3, 384, 384)
    print(f"Tabular Features:     {sample['tabular'].shape}")  # Expected: (192,)

    # Verification
    assert sample["dino_views"].shape == (4, 3, 518, 518), "Incorrect DINO tensor shape"
    assert sample["convnext_views"].shape == (
        4,
        3,
        384,
        384,
    ), "Incorrect ConvNeXt tensor shape"
    assert sample["tabular"].shape == (192,), "Incorrect tabular feature shape"
    assert "label" in sample, "Label missing from training sample"
    print("Data loading verification passed.")

    # ==========================================
    # 3. Feature Extraction Demonstration
    # ==========================================
    print("\n--- 3. Feature Extraction Demonstration ---")

    # Initialize Extractor (Loads DINOv2 and ConvNeXt)
    extractor = DualStreamExtractor()

    # Extract features for the training subset
    # We force recompute (load_cached_data=False) to demonstrate the extraction process
    print("Extracting features for training subset...")
    train_data = extractor.extract_features(
        Config.TRAIN_META_PATH, dataset_key="demo_train", load_cached_data=False
    )

    # Verify Output
    n_samples = len(dataset)
    # Embedding Dim = 1024 (DINO Large) + 1536 (ConvNeXt Large) = 2560
    expected_dim = 1024 + 1536

    print(f"Extracted Embeddings Shape: {train_data['embeddings'].shape}")

    assert train_data["embeddings"].shape == (n_samples, 4, expected_dim)
    assert train_data["tabular"].shape == (n_samples, 192)
    assert len(train_data["ids"]) == n_samples
    assert len(train_data["labels"]) == n_samples
    print("Feature extraction verification passed.")

    # ==========================================
    # 4. Preprocessing (Manifold Stabilization)
    # ==========================================
    print("\n--- 4. Preprocessing Demonstration ---")

    processor = ManifoldProcessor()

    # A. Fit & Transform on Training Data (Expanded View Strategy)
    # This expands N samples to 4N samples (1 per view) for robust training
    print("Fitting processor on training data (Expanded View)...")
    X_train, y_train, ids_train = processor.fit_transform_train(
        train_data, load_cache=False
    )

    print(f"Expanded Train Data Shape: {X_train.shape}")

    # Verify Expansion
    assert (
        X_train.shape[0] == n_samples * 4
    ), "Training data should be expanded by factor of 4"
    assert len(y_train) == n_samples * 4

    # B. Transform on Validation Data (Centroid View Strategy)
    # This averages views to N samples for stable inference
    # We use the same train_data here just to simulate validation input
    print("Transforming data for inference (Centroid View)...")
    X_val, y_val, ids_val = processor.transform_inference(
        train_data, prefix="demo_val", load_cache=False
    )

    print(f"Centroid Inference Data Shape: {X_val.shape}")

    # Verify Centroid
    assert (
        X_val.shape[0] == n_samples
    ), "Inference data should match original sample count"
    assert (
        X_val.shape[1] == X_train.shape[1]
    ), "PCA components mismatch between train and val"
    print("Preprocessing verification passed.")

    # ==========================================
    # 5. Classification Demonstration
    # ==========================================
    print("\n--- 5. Classification Demonstration ---")

    lda = LDAManager()

    # Train LDA on Expanded Features
    print("Training LDA classifier...")
    lda.train(X_train, y_train)

    # Predict on Centroid Features
    print("Predicting probabilities...")
    probs = lda.predict_proba(X_val)

    print(f"Probabilities Shape: {probs.shape}")

    # Verify Probabilities
    assert probs.shape == (n_samples, len(lda.classes_))
    assert np.all(probs >= 0.0) and np.all(
        probs <= 1.0
    ), "Probabilities out of range [0, 1]"
    print("Classification verification passed.")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\n--- 6. Test Set & Submission Generation ---")

    # 1. Extract Test Features
    print("Processing Test Set...")
    test_data = extractor.extract_features(
        Config.TEST_META_PATH, dataset_key="demo_test", load_cached_data=False
    )

    # 2. Transform Test Data (Inference Mode)
    X_test, _, ids_test = processor.transform_inference(
        test_data, prefix="demo_test", load_cache=False
    )

    # 3. Predict
    test_probs = lda.predict_proba(X_test)

    # 4. Format Submission
    # We must ensure the submission has all columns present in sample_submission.csv
    # Since we used a subset of training data, our model might not know all 99 classes.
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    all_classes = sample_sub.columns.tolist()
    if "id" in all_classes:
        all_classes.remove("id")

    # Create DataFrame with predictions available from the model
    pred_df = pd.DataFrame(test_probs, columns=lda.classes_)
    pred_df["id"] = ids_test

    # Construct final submission DataFrame with all required columns
    final_sub = pd.DataFrame()
    final_sub["id"] = ids_test

    # Fill columns: use prediction if available, else 0.0
    for cls in all_classes:
        if cls in pred_df.columns:
            final_sub[cls] = pred_df[cls]
        else:
            final_sub[cls] = 0.0

    # Save Submission
    final_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Final Validation
    saved_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Saved Submission Shape: {saved_df.shape}")

    assert (
        saved_df.shape[1] == 100
    ), "Submission must have 100 columns (id + 99 species)"
    assert "id" in saved_df.columns
    assert len(saved_df) == len(ids_test)

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
