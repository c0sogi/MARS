import os
import shutil
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.feature_extraction import FeatureExtractor
from library.data_processing import LeafDataProcessor
from library.modeling import StackedEnsemble


def main():
    # ==========================================
    # 1. Setup Demo Configuration & Data
    # ==========================================
    print("Initializing Demo Configuration...")

    # Define demo working directory
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config global settings for the demo to ensure speed
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.MODELS_DIR = os.path.join(DEMO_DIR, "models")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce complexity for speed
    Config.NUM_FOLDS = 2  # Minimum for CV
    Config.INNER_FOLDS = 2  # Minimum for inner CV
    Config.NUM_TRAIN_CENTROIDS = 2  # Reduce densification factor (default 9)

    # Create directories based on new config
    Config.setup_directories()

    # Create Subset Metadata
    # We filter the original metadata to a very small subset to ensure execution < 5 mins
    print("Creating subset metadata for demo...")
    meta_out_dir = os.path.join(DEMO_DIR, "metadata")
    os.makedirs(meta_out_dir, exist_ok=True)

    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Select top 3 classes to ensure enough samples for StratifiedKFold
    top_classes = orig_train["species"].value_counts().head(3).index.tolist()

    # Take 4 samples per class for Train, 2 for Val
    # This ensures ~6 samples per class total, enough for 2-fold outer + 2-fold inner CV
    demo_train = (
        orig_train[orig_train["species"].isin(top_classes)].groupby("species").head(4)
    )
    demo_val = (
        orig_val[orig_val["species"].isin(top_classes)].groupby("species").head(2)
    )
    demo_test = orig_test.head(5)

    # Save subset metadata
    path_train = os.path.join(meta_out_dir, "train.csv")
    path_val = os.path.join(meta_out_dir, "val.csv")
    path_test = os.path.join(meta_out_dir, "test.csv")

    demo_train.to_csv(path_train, index=False)
    demo_val.to_csv(path_val, index=False)
    demo_test.to_csv(path_test, index=False)

    # Point Config to these new files
    Config.METADATA_TRAIN = path_train
    Config.METADATA_VAL = path_val
    Config.METADATA_TEST = path_test

    print(
        f"Demo Data: {len(demo_train)} Train, {len(demo_val)} Val, {len(demo_test)} Test samples."
    )

    # ==========================================
    # 2. Feature Extraction
    # ==========================================
    print("\n--- Feature Extraction ---")
    # Instantiate extractor
    extractor = FeatureExtractor()

    # Extract features (force reload to ignore any pre-existing cache)
    # This will run DINOv2 and ConvNeXt on the subset images
    raw_data = extractor.extract_all(load_cached_data=False)

    # Verify dimensions
    # DINOv2 Large = 1024, ConvNeXt Large = 1536
    # Shape: (N, 36, D)
    assert raw_data["train_dino"].shape[1:] == (
        36,
        1024,
    ), "Incorrect DINO feature shape"
    assert raw_data["train_conv"].shape[1:] == (
        36,
        1536,
    ), "Incorrect ConvNeXt feature shape"
    assert len(raw_data["train_ids"]) == len(
        demo_train
    ), "Mismatch in training IDs count"

    # ==========================================
    # 3. Data Processing (Densification & PCA)
    # ==========================================
    print("\n--- Data Processing ---")
    # Initialize processor (loads the raw data we just extracted from cache)
    processor = LeafDataProcessor(load_raw_cache=True)

    # Get processed data for Fold 0
    # This performs hyper-densification (creating multiple centroids per image)
    # and dimensionality reduction (PCA)
    fold_data = processor.get_fold_data(fold_idx=0, load_cache=False)

    # Verify processed data
    print("Verifying Fold 0 data...")
    X_d_tr = fold_data["X_dino_train"]
    y_tr = fold_data["y_train"]

    # Check densification: N_train_samples * NUM_TRAIN_CENTROIDS
    assert X_d_tr.shape[0] == y_tr.shape[0], "Feature and label count mismatch"
    assert fold_data["classes"].shape[0] == 3, "Should have exactly 3 classes"

    # Check PCA reduction
    # With very few samples, PCA components = min(n_samples, variance_threshold)
    assert X_d_tr.shape[1] < 1024, "PCA did not reduce dimensions"

    # ==========================================
    # 4. Modeling (Stacked Ensemble)
    # ==========================================
    print("\n--- Modeling ---")
    ensemble = StackedEnsemble()

    # Fit the ensemble
    # This runs Inner CV (2-fold) to generate OOF logits, trains Meta-Learner,
    # and retrains base experts.
    ensemble.fit(
        fold_data["X_dino_train"],
        fold_data["X_conv_train"],
        fold_data["X_tab_train"],
        fold_data["y_train"],
        fold_data["ids_train"],
    )

    # ==========================================
    # 5. Inference & Evaluation
    # ==========================================
    print("\n--- Inference ---")

    # Predict on Validation
    val_probs = ensemble.predict_proba(
        fold_data["X_dino_val"], fold_data["X_conv_val"], fold_data["X_tab_val"]
    )

    # Calculate Metric
    # Here y_val are integer indices from LabelEncoder
    val_loss = calculate_log_loss(fold_data["y_val"], val_probs)
    print(f"Validation Log Loss (Fold 0): {val_loss:.6f}")

    # Predict on Test
    test_probs = ensemble.predict_proba(
        fold_data["X_dino_test"], fold_data["X_conv_test"], fold_data["X_tab_test"]
    )

    # Verify probabilities
    # Check range [0, 1]
    assert np.all(test_probs >= 0) and np.all(
        test_probs <= 1
    ), "Probabilities out of range"
    # Check row sums (should be approx 1)
    row_sums = np.sum(test_probs, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\n--- Generating Submission ---")

    # Create DataFrame
    classes = fold_data["classes"]
    submission_df = pd.DataFrame(test_probs, columns=classes)

    # Add ID column
    submission_df.insert(0, "id", fold_data["ids_test"])

    # Save
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)

    # Final Verification
    if os.path.exists(Config.SUBMISSION_FILE):
        print(f"Submission successfully saved to: {Config.SUBMISSION_FILE}")
        print("First 5 rows:")
        print(submission_df.head())
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
