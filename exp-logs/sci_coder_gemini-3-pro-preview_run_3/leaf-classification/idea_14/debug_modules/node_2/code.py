import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")

# Import from provided library files
from library.configuration import Config
from library.utilities import seed_everything
from library.vision_extractor import (
    DualStreamExtractor,
    LeafRotationDataset,
    extract_rotational_features,
)
from library.topology_manager import TopologyManager
from library.custom_pipeline import LeafSpeciesPipeline
from library.engine import run_cross_validation, generate_submission


def demo_vision_components():
    print("\n=== Demonstrating Vision Components ===")

    # 1. Test LeafRotationDataset
    print("1. Testing LeafRotationDataset...")
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    # Use a tiny subset
    df_subset = df_train.iloc[:2]

    dataset = LeafRotationDataset(df_subset, Config.INPUT_DIR)

    # Get first item
    # Expected shape: (12, 3, 224, 224)
    views_tensor = dataset[0]
    print(f"   Dataset item shape: {views_tensor.shape}")

    assert views_tensor.shape == (
        12,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Expected (12, 3, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE}), got {views_tensor.shape}"

    # 2. Test DualStreamExtractor Model
    print("2. Testing DualStreamExtractor Model (Forward Pass)...")
    device = torch.device(Config.DEVICE)

    model = DualStreamExtractor()
    model.to(device)
    model.eval()

    # Create dummy input: Batch of 2 images, each with 3 channels, 224x224
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)

    with torch.no_grad():
        features = model(dummy_input)

    print(f"   Extracted features shape: {features.shape}")

    # DINOv2 Large (1024) + ConvNeXt Large (1536) = 2560
    expected_dim = 1024 + 1536
    assert features.shape == (
        2,
        expected_dim,
    ), f"Expected (2, {expected_dim}), got {features.shape}"

    # 3. Test extract_rotational_features function
    print("3. Testing extract_rotational_features function...")
    # We use a limit of 2 samples to be fast
    # This will run the full inference loop on these 2 samples (2 * 12 = 24 views)
    features_array = extract_rotational_features(
        df_train,
        subset_name="train_demo",
        load_cached_data=False,  # Force computation
        limit=2,
    )

    print(f"   Full extraction output shape: {features_array.shape}")
    assert features_array.shape == (
        2,
        12,
        expected_dim,
    ), f"Expected (2, 12, {expected_dim}), got {features_array.shape}"


def demo_topology_manager():
    print("\n=== Demonstrating Topology Manager ===")

    tm = TopologyManager()
    limit = 6  # Use enough samples to allow PCA to run without issues

    # 1. Densified Training Data
    print(f"1. Getting Densified Training Data (Limit={limit})...")
    # This generates 3 centroids per image -> 3 * 6 = 18 samples
    X_img, X_tab, y, ids = tm.get_densified_train_data(
        load_cached_data=False, limit=limit  # Force re-computation to verify logic
    )

    print(f"   X_img shape: {X_img.shape}")
    print(f"   X_tab shape: {X_tab.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   ids shape: {ids.shape}")

    expected_samples = limit * 3
    expected_dim = 1024 + 1536

    assert X_img.shape == (expected_samples, expected_dim)
    assert X_tab.shape == (expected_samples, 192)
    assert y.shape == (expected_samples,)
    assert ids.shape == (expected_samples,)

    # Verify ID replication logic (A, B, C structure)
    # ids should be [id1, id2, ..., id1, id2, ..., id1, id2, ...]
    # Check if first third matches second third
    first_third = ids[:limit]
    second_third = ids[limit : 2 * limit]
    assert np.array_equal(
        first_third, second_third
    ), "IDs were not replicated correctly for densification."

    # 2. Canonical Inference Data
    print(f"2. Getting Canonical Inference Data (Limit={limit})...")
    # This generates 1 centroid per image -> 6 samples
    X_img_val, X_tab_val, y_val, ids_val = tm.get_canonical_inference_data(
        subset="val", load_cached_data=False, limit=limit
    )

    print(f"   Val X_img shape: {X_img_val.shape}")

    assert X_img_val.shape == (limit, expected_dim)
    assert X_tab_val.shape == (limit, 192)

    return X_img, X_tab, y, X_img_val, X_tab_val


def demo_pipeline(X_img, X_tab, y, X_img_val, X_tab_val):
    print("\n=== Demonstrating Custom Pipeline ===")

    pipeline = LeafSpeciesPipeline(
        dino_dim=1024, pca_variance=0.99, random_state=Config.SEED
    )

    print("1. Fitting Pipeline...")
    pipeline.fit(X_img, X_tab, y)

    print("2. Predicting Probabilities...")
    probs = pipeline.predict_proba(X_img_val, X_tab_val)

    print(f"   Probabilities shape: {probs.shape}")

    # Check shape: (n_val_samples, n_classes_in_subset)
    # Note: y might not contain all 99 classes if we only took top 6 rows.
    n_unique_classes = len(np.unique(y))
    assert probs.shape == (X_img_val.shape[0], n_unique_classes)

    # Check range
    assert np.all(probs >= 0) and np.all(probs <= 1)

    print("   Pipeline verification successful.")


def demo_engine():
    print("\n=== Demonstrating Engine (Full Workflow) ===")

    # The engine functions run_cross_validation and generate_submission
    # have a 'debug' flag which limits samples internally.

    print("1. Running Cross-Validation (Debug Mode)...")
    run_cross_validation(debug=True)

    # Check if model files were created
    # Debug mode in engine uses 2 folds.
    model_path_0 = os.path.join(Config.WORKING_DIR, "model_fold_0.joblib")
    assert os.path.exists(model_path_0), f"Model file {model_path_0} was not created."

    print("2. Generating Submission (Debug Mode)...")
    generate_submission(debug=True)

    # Check submission file
    sub_path = Config.SUBMISSION_PATH
    assert os.path.exists(sub_path), f"Submission file {sub_path} was not created."

    df_sub = pd.read_csv(sub_path)
    print(f"   Submission file loaded. Shape: {df_sub.shape}")

    # Sample submission has 100 columns (id + 99 species)
    assert df_sub.shape[1] == 100, f"Expected 100 columns, got {df_sub.shape[1]}"
    print("   Engine verification successful.")


if __name__ == "__main__":
    # Setup
    seed_everything(42)

    # Modify Config for Demo Speed
    # We can modify class attributes directly
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 6  # Small enough for speed

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    try:
        # Run Demos
        demo_vision_components()

        # Get data for pipeline demo from topology demo
        X_img, X_tab, y, X_img_val, X_tab_val = demo_topology_manager()

        demo_pipeline(X_img, X_tab, y, X_img_val, X_tab_val)

        demo_engine()

        print("\nAll demonstrations completed successfully!")

    except Exception as e:
        print(f"\n\n!!! An error occurred during demonstration !!!")
        print(e)
        import traceback

        traceback.print_exc()
        sys.exit(1)
