import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings
import torch

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_log_loss, save_submission
from library.data_manager import DataManager
from library.modeling import OrthogonalLDAPipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def setup_demo_environment():
    """
    Sets up a temporary demo environment with a subset of data to ensure
    fast execution for demonstration purposes.
    """
    print("Setting up demo environment...")

    # Define demo directories
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample a small subset (stratified sampling not strictly necessary for demo code check, but good practice)
    # We take 20 train, 5 val, 5 test samples
    demo_train = orig_train.head(20).copy()
    demo_val = orig_val.head(5).copy()
    demo_test = orig_test.head(5).copy()

    # Save demo metadata
    demo_train_path = os.path.join(demo_dir, "train.csv")
    demo_val_path = os.path.join(demo_dir, "val.csv")
    demo_test_path = os.path.join(demo_dir, "test.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    print(
        f"Created demo metadata: Train={len(demo_train)}, Val={len(demo_val)}, Test={len(demo_test)}"
    )

    # ==========================================
    # Monkey-patch Config to use demo paths
    # ==========================================
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_val_path
    Config.TEST_METADATA_PATH = demo_test_path
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")

    # Update cache paths to point to demo directory
    Config.CACHE_TRAIN_IMG_FEATURES = os.path.join(demo_dir, "train_img_features.npy")
    Config.CACHE_TRAIN_TAB_FEATURES = os.path.join(demo_dir, "train_tab_features.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(demo_dir, "train_labels.npy")
    Config.CACHE_TRAIN_IDS = os.path.join(demo_dir, "train_ids.npy")

    Config.CACHE_TEST_IMG_FEATURES = os.path.join(demo_dir, "test_img_features.npy")
    Config.CACHE_TEST_TAB_FEATURES = os.path.join(demo_dir, "test_tab_features.npy")
    Config.CACHE_TEST_IDS = os.path.join(demo_dir, "test_ids.npy")

    # Reduce batch size for demo to ensure it runs smoothly even if memory was tight (though A100 is fine)
    Config.BATCH_SIZE = 8

    return demo_train, demo_val, demo_test


def verify_densification_logic(raw_data, densified_data):
    """
    Verifies that Manifold Densification (N -> 3N) logic is correct.
    """
    print("Verifying Manifold Densification logic...")

    img_raw, tab_raw, lbl_raw, ids_raw = raw_data

    # Check 1: Sample Size Multiplier
    n_raw = len(lbl_raw)
    n_dens = len(densified_data["labels"])

    assert (
        n_dens == 3 * n_raw
    ), f"Densification failed: Expected {3*n_raw} samples, got {n_dens}"

    # Check 2: Feature Dimensions
    # DINO dim = 1024, ConvNeXt dim = 1536
    assert (
        densified_data["dino"].shape[1] == 1024
    ), f"Incorrect DINO dimension: {densified_data['dino'].shape[1]}"
    assert (
        densified_data["convnext"].shape[1] == 1536
    ), f"Incorrect ConvNeXt dimension: {densified_data['convnext'].shape[1]}"

    # Check 3: ID Replication
    # The first N IDs in densified should match the raw IDs (corresponding to Centroid A)
    # Note: The implementation concatenates centroids [A, B, C], so the first block is A.
    # However, DataManager.create_orthogonal_centroids tiles IDs: np.tile(ids, num_groups)
    # This means [ID1, ID2, ..., ID1, ID2, ...]

    # Let's verify the tiling structure
    ids_densified = densified_data["ids"]
    assert np.array_equal(
        ids_densified[:n_raw], ids_raw
    ), "First block of densified IDs does not match raw IDs"
    assert np.array_equal(
        ids_densified[n_raw : 2 * n_raw], ids_raw
    ), "Second block of densified IDs does not match raw IDs"

    print("Densification logic verified successfully.")


def main():
    # 1. Reproducibility
    seed_everything(42)

    # 2. Setup Demo Environment
    setup_demo_environment()

    # 3. Initialize Data Manager
    dm = DataManager()

    # 4. Load Raw Data (Feature Extraction)
    # This will run the DINOv2 and ConvNeXt models on the small subset
    print("\n--- Step 1: Feature Extraction ---")
    print("Extracting features (this involves loading large models, please wait)...")

    # Force re-extraction to demonstrate the pipeline (load_cached_data=False)
    # Note: In a real run, you'd likely use True.
    (train_img, train_tab, train_lbl, train_ids), (test_img, test_tab, test_ids) = (
        dm.load_raw_data(load_cached_data=False)
    )

    print(f"Extracted Train Shape: {train_img.shape} (N, 12, 2560)")
    print(f"Extracted Test Shape:  {test_img.shape} (N, 12, 2560)")

    # 5. Manifold Densification
    print("\n--- Step 2: Manifold Densification ---")
    train_densified = dm.create_orthogonal_centroids(
        train_img, train_tab, labels=train_lbl, ids=train_ids
    )

    # Verify the logic
    verify_densification_logic(
        (train_img, train_tab, train_lbl, train_ids), train_densified
    )

    # Densify test data as well
    test_densified = dm.create_orthogonal_centroids(test_img, test_tab, ids=test_ids)

    # 6. Modeling (Orthogonal LDA Pipeline)
    print("\n--- Step 3: Model Training & Inference ---")

    # Encode labels
    unique_classes = np.unique(train_lbl)
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    y_train_encoded = np.array([class_to_idx[l] for l in train_densified["labels"]])

    # Initialize Pipeline
    # Using a slightly lower variance for PCA in demo to ensure it runs even with very few samples
    # (PCA cannot find more components than min(n_samples, n_features))
    # With 20 samples * 3 = 60 samples, keeping 0.99 variance is fine, but let's be safe.
    pipeline = OrthogonalLDAPipeline(pca_variance=0.99, random_state=42)

    # Fit
    print("Fitting OrthogonalLDAPipeline...")
    pipeline.fit(
        train_densified["dino"],
        train_densified["convnext"],
        train_densified["tabular"],
        y_train_encoded,
    )

    # Predict on Test (Densified)
    print("Predicting on Test set...")
    test_probs_dens = pipeline.predict_proba(
        test_densified["dino"], test_densified["convnext"], test_densified["tabular"]
    )

    # Check output shape
    assert test_probs_dens.shape == (
        len(test_densified["ids"]),
        len(unique_classes),
    ), "Prediction shape mismatch"

    # 7. Aggregation & Submission
    print("\n--- Step 4: Aggregation & Submission ---")

    # Create DataFrame for aggregation
    # We need to map the probability columns to the specific classes present in our training subset
    # Note: In the real task, we need to output columns for ALL 99 classes.
    # For this demo, we will pad the missing classes with zeros to match submission format requirements.

    # Get all possible classes from the original sample_submission or metadata description
    # Here we simulate reading the full class list from the sample submission provided in description
    # For the demo, we'll just use the classes found in the subset + dummy classes to show logic

    # Aggregate (Mean across centroids for each image ID)
    df_pred = pd.DataFrame(test_probs_dens, columns=unique_classes)
    df_pred["id"] = test_densified["ids"]
    df_agg = df_pred.groupby("id").mean().reset_index()

    print("Aggregated predictions shape:", df_agg.shape)

    # Verify values are in [0, 1]
    assert df_agg.drop("id", axis=1).max().max() <= 1.0 + 1e-6
    assert df_agg.drop("id", axis=1).min().min() >= 0.0 - 1e-6

    # Save Submission
    # We pass the subset classes. In a real scenario, you'd pass the full list of 99 classes.
    save_submission(
        df_agg["id"].values,
        df_agg[unique_classes].values,
        list(unique_classes),
        output_path=Config.SUBMISSION_PATH,
    )

    # Verify file creation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # 8. Metric Calculation (Self-Check on Train Data for Demo)
    # We predict on the training set just to verify the metric function works
    print("\n--- Step 5: Metric Verification ---")
    train_probs_dens = pipeline.predict_proba(
        train_densified["dino"], train_densified["convnext"], train_densified["tabular"]
    )

    # Calculate Log Loss
    # We use the densified labels directly for this check
    loss = calculate_log_loss(y_train_encoded, train_probs_dens)
    print(f"Training Log Loss (Densified): {loss:.4f}")

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
