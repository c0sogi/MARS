import os
import shutil
import numpy as np
import pandas as pd
import logging

# Import provided library modules
from library import (
    config,
    utils,
    feature_extractor,
    data_manager,
    pipeline_builder,
    execution_engine,
)


def run_demo():
    # =========================================================================
    # 1. Setup and Configuration Overrides
    # =========================================================================
    print(">>> [1/6] Setting up demonstration configuration...")
    utils.seed_everything(42)

    # Define a temporary directory for this demo run
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override global config parameters to optimize for speed
    config.WORKING_DIR = DEMO_DIR
    config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Reduce workload: 2 folds, small batch size
    config.N_FOLDS = 2
    config.BATCH_SIZE = 4

    # Silence third-party libraries
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("timm").setLevel(logging.ERROR)

    # =========================================================================
    # 2. Create Subset Metadata (Mock Data)
    # =========================================================================
    print(">>> [2/6] Creating stratified subset metadata for fast execution...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Create a stratified training subset (12 samples: 6 Class A, 6 Class B)
    # We overwrite the species column to ensure we have exactly 2 classes for the 2-fold CV
    demo_train = orig_train.iloc[0:12].copy()
    demo_train["species"] = ["Species_A"] * 6 + ["Species_B"] * 6

    # Create a validation subset (4 samples, distinct from train)
    demo_val = orig_train.iloc[12:16].copy()
    demo_val["species"] = ["Species_A"] * 2 + ["Species_B"] * 2

    # Create a test subset
    demo_test = orig_test.iloc[0:4].copy()

    # Save these subsets to the demo directory
    demo_meta_dir = os.path.join(DEMO_DIR, "metadata")
    os.makedirs(demo_meta_dir, exist_ok=True)

    path_train = os.path.join(demo_meta_dir, "train.csv")
    path_val = os.path.join(demo_meta_dir, "val.csv")
    path_test = os.path.join(demo_meta_dir, "test.csv")

    demo_train.to_csv(path_train, index=False)
    demo_val.to_csv(path_val, index=False)
    demo_test.to_csv(path_test, index=False)

    # Point config to these new files so library modules use them
    config.TRAIN_METADATA_PATH = path_train
    config.VAL_METADATA_PATH = path_val
    config.TEST_METADATA_PATH = path_test

    print(f"    Subset metadata created at {demo_meta_dir}")

    # =========================================================================
    # 3. Demonstrate Feature Extractor
    # =========================================================================
    print(">>> [3/6] Demonstrating LeafFeatureExtractor...")
    extractor = feature_extractor.LeafFeatureExtractor()

    # Test 1: Single image preprocessing
    sample_path = os.path.join(config.INPUT_DIR, demo_train.iloc[0]["file_path"])
    pil_img = extractor.preprocess_image(sample_path)

    assert (
        pil_img.size == config.IMG_SIZE
    ), f"Preprocessing failed. Expected size {config.IMG_SIZE}, got {pil_img.size}"

    # Test 2: Multi-view generation
    views = extractor.get_12_views(pil_img)
    # Expected shape: [12, 3, H, W]
    assert views.shape == (
        12,
        3,
        config.IMG_SIZE[1],
        config.IMG_SIZE[0],
    ), f"View generation failed. Got shape {views.shape}"

    # Test 3: Batch extraction
    batch_paths = [
        os.path.join(config.INPUT_DIR, p) for p in demo_train.iloc[:2]["file_path"]
    ]
    dino_feats, conv_feats = extractor.extract_batch(batch_paths)

    # Verify feature shapes: [Batch, 12, Dim]
    # DINOv2-large dim: 1024, ConvNeXt-large dim: 1536
    assert dino_feats.shape == (2, 12, 1024), f"DINO shape mismatch: {dino_feats.shape}"
    assert conv_feats.shape == (
        2,
        12,
        1536,
    ), f"ConvNeXt shape mismatch: {conv_feats.shape}"
    print("    Feature extractor verified.")

    # =========================================================================
    # 4. Demonstrate Data Manager
    # =========================================================================
    print(">>> [4/6] Demonstrating LeafDataManager...")
    dm = data_manager.LeafDataManager()

    # Process 'train' dataset (Extract -> Centroids -> Densify)
    # load_cached_data=False forces fresh extraction
    train_data = dm.get_dataset("train", load_cached_data=False)

    # Verify dictionary structure
    required_keys = {"dino", "conv", "tabular", "ids", "y"}
    assert required_keys.issubset(train_data.keys()), "Data Manager missing keys."

    # Verify Densification Logic
    # 12 input images -> 3 centroids each -> 36 rows
    n_images = 12
    n_densified = n_images * 3

    assert (
        len(train_data["ids"]) == n_densified
    ), f"Densification error. Expected {n_densified} rows, got {len(train_data['ids'])}"
    assert train_data["dino"].shape[0] == n_densified
    assert train_data["conv"].shape[0] == n_densified
    assert train_data["tabular"].shape[0] == n_densified

    print("    Data Manager verified (Extraction & Densification successful).")

    # =========================================================================
    # 5. Demonstrate Pipeline Builder
    # =========================================================================
    print(">>> [5/6] Demonstrating Pipeline Builder...")

    dino_dim = train_data["dino"].shape[1]
    conv_dim = train_data["conv"].shape[1]
    tab_dim = train_data["tabular"].shape[1]

    pipeline = pipeline_builder.build_selective_pipeline(dino_dim, conv_dim, tab_dim)

    # Dry-run fit to ensure pipeline graph is valid
    X_mock = np.hstack([train_data["dino"], train_data["conv"], train_data["tabular"]])
    y_mock = train_data["y"]

    pipeline.fit(X_mock, y_mock)
    print("    Pipeline built and fitted successfully.")

    # =========================================================================
    # 6. Demonstrate Execution Engine (Full Workflow)
    # =========================================================================
    print(">>> [6/6] Demonstrating Execution Engine...")

    # A. Train Ensemble
    # This runs Stratified K-Fold (k=2) on the combined train+val data
    # We enable caching so it reuses the 'train' features we extracted in step 4
    print("    Starting Ensemble Training...")
    execution_engine.train_ensemble(load_cached_data=True)

    # Verify model artifacts
    models_dir = os.path.join(config.WORKING_DIR, "models")
    assert os.path.exists(
        os.path.join(models_dir, "pipeline_fold_0.pkl")
    ), "Fold 0 model missing."
    assert os.path.exists(
        os.path.join(models_dir, "classes.pkl")
    ), "Classes file missing."

    # B. Predict Submission
    # This processes the test set and generates the submission file
    print("    Generating Submission...")
    execution_engine.predict_submission(load_cached_data=False)

    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not created."

    # Verify submission content
    df_sub = pd.read_csv(submission_path)
    print(f"    Submission generated: {df_sub.shape}")

    # Check that we have the ID column and prediction columns
    assert "id" in df_sub.columns
    # Check that missing columns (since we only trained on 2 classes) were filled
    assert (
        df_sub.shape[1] == 100
    ), f"Expected 100 columns (id + 99 classes), got {df_sub.shape[1]}"

    print("\n>>> Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
