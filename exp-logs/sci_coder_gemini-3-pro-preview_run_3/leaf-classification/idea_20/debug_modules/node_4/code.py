import os
import shutil
import pandas as pd
import numpy as np
import torch
from library.config import Config
from library.utils import seed_everything
from library.feature_extraction import FeatureExtractor
from library.data_processing import ManifoldDensifier
from library.modeling import ModelingPipeline, DualStreamLDA


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo run.
    Creates a subset of the metadata to speed up feature extraction.
    """
    print(">>> Setting up demo environment...")

    # Define demo directories
    demo_dir = "./working/demo_execution"
    demo_data_dir = os.path.join(demo_dir, "data")
    os.makedirs(demo_data_dir, exist_ok=True)

    # Override Config global settings for the demo
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    Config.N_FOLDS = 2  # Reduce folds for speed

    # Load original metadata
    # We use the original paths initially to read the data
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Create a small subset
    # We need enough samples for 2-fold CV and ensuring we have valid classes
    # Let's pick 3 classes and take 6 samples each for training (18 total)
    # and 5 random samples for testing.
    selected_classes = orig_train["species"].value_counts().head(3).index.tolist()
    subset_train = (
        orig_train[orig_train["species"].isin(selected_classes)]
        .groupby("species")
        .head(6)
        .reset_index(drop=True)
    )
    subset_test = orig_test.head(5).reset_index(drop=True)

    print(
        f"Subset Training Data: {len(subset_train)} samples (Classes: {selected_classes})"
    )
    print(f"Subset Test Data: {len(subset_test)} samples")

    # Save subset metadata
    demo_train_path = os.path.join(demo_data_dir, "train_subset.csv")
    demo_val_path = os.path.join(
        demo_data_dir, "val_subset.csv"
    )  # Just reuse train for val meta placeholder
    demo_test_path = os.path.join(demo_data_dir, "test_subset.csv")

    subset_train.to_csv(demo_train_path, index=False)
    subset_train.to_csv(
        demo_val_path, index=False
    )  # In the pipeline, train+val are concatenated
    subset_test.to_csv(demo_test_path, index=False)

    # Update Config paths to point to these subsets
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_val_path
    Config.TEST_METADATA_PATH = demo_test_path

    # Re-run setup to ensure directories are created based on new Config
    Config.setup()

    return subset_train, subset_test


def verify_feature_extraction():
    """
    Demonstrates and verifies FeatureExtractor.
    """
    print("\n>>> Running Feature Extraction (Demo Subset)...")
    extractor = FeatureExtractor()

    # Force run without cache to demonstrate the extraction logic
    extractor.run(load_cached_data=False)

    # Verify output files exist
    expected_files = [
        "train_densified_img_features.npy",
        "train_densified_tab_features.npy",
        "train_densified_ids.npy",
        "train_densified_labels.npy",
        "test_densified_img_features.npy",
    ]

    for f in expected_files:
        path = os.path.join(Config.WORKING_DIR, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature extraction failed to create {f}")

    # Verify shapes
    # Note: In the pipeline, train and val metadata are concatenated.
    # We used the same subset for both train and val paths in setup_demo_environment.
    # So the total training samples processed will be len(subset_train) * 2.
    train_img = np.load(Config.get_cache_path("train_img_features"))

    # Expected shape: (N_samples, 12_views, 2560_features)
    # 2560 = 1024 (DINO) + 1536 (ConvNeXt)
    assert (
        train_img.ndim == 3
    ), f"Expected 3D array for image features, got {train_img.ndim}"
    assert train_img.shape[1] == 12, f"Expected 12 views, got {train_img.shape[1]}"
    assert (
        train_img.shape[2] == 2560
    ), f"Expected 2560 features, got {train_img.shape[2]}"

    print("Feature extraction verification passed.")


def verify_data_processing():
    """
    Demonstrates and verifies ManifoldDensifier.
    """
    print("\n>>> Running Manifold Densification...")
    densifier = ManifoldDensifier()

    # This will load the features generated in the previous step and process them
    train_data, test_data = densifier.run(load_cached_data=False)

    # Verify Densification Logic
    # The densifier creates 3 centroids per image.
    # If input had N images, output should have 3*N images.

    # Check Train
    raw_train_ids = np.load(Config.get_cache_path("train_ids"))
    densified_train_ids = train_data["ids"]

    assert len(densified_train_ids) == 3 * len(
        raw_train_ids
    ), f"Densification should triple dataset size. Raw: {len(raw_train_ids)}, Densified: {len(densified_train_ids)}"

    # Check Test
    raw_test_ids = np.load(Config.get_cache_path("test_ids"))
    densified_test_ids = test_data["ids"]

    assert len(densified_test_ids) == 3 * len(
        raw_test_ids
    ), f"Densification should triple test set size."

    # Check Feature Consistency
    # Tabular features should be replicated
    assert train_data["tab"].shape[0] == len(densified_train_ids)
    assert train_data["tab"].shape[1] == 192  # 64 margin + 64 shape + 64 texture

    print("Manifold densification verification passed.")
    return train_data, test_data


def verify_modeling(train_data, test_data):
    """
    Demonstrates and verifies ModelingPipeline and DualStreamLDA.
    """
    print("\n>>> Running Modeling Pipeline...")

    pipeline = ModelingPipeline()

    # 1. Run Training
    # This will train DualStreamLDA on the densified data using 2-Fold CV
    pipeline.run_training(train_data)

    # Verify models were saved
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model for fold {fold} was not saved.")

    # 2. Run Inference
    # This performs Full-Manifold Test-Time Aggregation
    pipeline.run_inference(test_data)

    # Verify Submission
    submission_path = Config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(submission_path)

    # Check dimensions
    # We processed 'subset_test' which has 5 rows.
    expected_rows = 5
    # Columns: 'id' + 99 species = 100 columns (assuming label encoder saw all classes,
    # but here we only trained on a subset of classes.
    # The ModelingPipeline uses LabelEncoder on the training labels.
    # In our subset, we only selected 3 classes. So the submission will only have columns for those 3 classes + id.)
    # Note: The provided format_submission function creates columns based on class_names passed to it.

    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"
    assert "id" in df_sub.columns, "Submission missing 'id' column"

    # Verify probabilities
    # Drop ID column
    probs = df_sub.drop(columns=["id"]).values

    # Check range [0, 1] (allowing for float precision)
    assert np.all(probs >= 0.0) and np.all(
        probs <= 1.0
    ), "Probabilities out of range [0, 1]"

    # Check that we have predictions for the classes we trained on
    print(f"Submission shape: {df_sub.shape}")
    print("Sample submission rows:")
    print(df_sub.head())

    print("Modeling pipeline verification passed.")


def verify_dual_stream_lda_component():
    """
    Unit test style verification for the custom DualStreamLDA estimator.
    """
    print("\n>>> Verifying DualStreamLDA Component Logic...")

    # Create synthetic data
    # DINO dim = 1024, ConvNeXt dim = 1536 -> Total 2560
    # Tabular dim = 192
    N = 50
    X_img = np.random.rand(N, 2560).astype(np.float32)
    X_tab = np.random.rand(N, 192).astype(np.float32)
    y = np.random.randint(0, 3, size=N)

    model = DualStreamLDA(
        pca_variance=0.95
    )  # Use lower variance to ensure reduction happens

    # Test Fit
    model.fit(X_img, X_tab, y)

    # Check internal state
    assert model.pca_dino is not None, "PCA DINO not initialized"
    assert model.pca_conv is not None, "PCA ConvNeXt not initialized"
    assert model.lda is not None, "LDA not initialized"

    # Test Predict Proba
    probs = model.predict_proba(X_img, X_tab)
    assert probs.shape == (N, 3), f"Expected prob shape ({N}, 3), got {probs.shape}"

    print("DualStreamLDA component verification passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Verify Feature Extraction
        verify_feature_extraction()

        # 3. Verify Data Processing
        train_data, test_data = verify_data_processing()

        # 4. Verify Component Logic
        verify_dual_stream_lda_component()

        # 5. Verify Full Modeling Pipeline
        verify_modeling(train_data, test_data)

        print("\n>>> All demonstrations and verifications completed successfully.")

    except Exception as e:
        print(f"\n!!! Error occurred during demonstration: {e}")
        raise e
