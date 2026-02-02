import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline

# Import library components
from library.config import Config
from library.utils import seed_everything, save_submission
from library.image_loader import LeafDataset
from library.feature_extractor import process_split
from library.topology_manager import TopologyTransformer
from library.model_factory import ModelFactory


def run_demo():
    print("=== Starting Library Usage Demo ===\n")

    # 1. Setup
    seed_everything(42)
    # Use a very small subset for speed
    DEMO_LIMIT = 4
    print(
        f"Configuration: Processing subset of {DEMO_LIMIT} samples for demonstration."
    )

    # Ensure metadata exists (provided by environment)
    if not os.path.exists(Config.TRAIN_METADATA):
        raise FileNotFoundError(f"Metadata file not found at {Config.TRAIN_METADATA}")

    # Load metadata to get real file paths for testing
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    sample_paths = [
        os.path.join(Config.INPUT_DIR, p)
        for p in df_train["file_path"].head(DEMO_LIMIT).tolist()
    ]

    # ==========================================
    # 2. Demonstrate Image Loader
    # ==========================================
    print("\n[Demo] Testing LeafDataset...")
    dataset = LeafDataset(sample_paths, rotation_angle=0)

    # Verify length
    assert (
        len(dataset) == DEMO_LIMIT
    ), f"Dataset length mismatch. Expected {DEMO_LIMIT}, got {len(dataset)}"

    # Verify item shape and type
    img_tensor = dataset[0]
    print(f"  Image Tensor Shape: {img_tensor.shape}")

    assert isinstance(img_tensor, torch.Tensor), "Dataset should return a torch.Tensor"
    assert img_tensor.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected shape (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img_tensor.shape}"

    # Check normalization (values should likely be small floats, not 0-255)
    assert (
        img_tensor.max() <= 5.0 and img_tensor.min() >= -5.0
    ), "Image data does not appear to be normalized (values out of expected range for ImageNet norm)."
    print("  LeafDataset validation passed.")

    # ==========================================
    # 3. Demonstrate Feature Extraction
    # ==========================================
    print("\n[Demo] Testing Feature Extraction (DualStreamExtractor)...")
    # process_split handles loading metadata, calling the extractor, and returning a dict
    # We use limit=DEMO_LIMIT to avoid processing the whole dataset

    # Note: This step runs the heavy DL models. On GPU it's fast.
    train_data = process_split("train", load_cached_data=False, limit=DEMO_LIMIT)

    # Verify keys
    required_keys = ["dino", "convnext", "tabular", "ids", "labels"]
    for k in required_keys:
        assert k in train_data, f"Missing key '{k}' in extraction result."

    # Verify shapes
    # Visual features: (N, Views, Dim)
    n_samples = DEMO_LIMIT
    n_views = Config.NUM_VIEWS

    dino_shape = train_data["dino"].shape
    conv_shape = train_data["convnext"].shape
    tab_shape = train_data["tabular"].shape

    print(f"  DINOv2 Features Shape: {dino_shape}")
    print(f"  ConvNeXt Features Shape: {conv_shape}")
    print(f"  Tabular Features Shape: {tab_shape}")

    assert (
        dino_shape[0] == n_samples and dino_shape[1] == n_views
    ), f"DINO shape mismatch. Expected ({n_samples}, {n_views}, D), got {dino_shape}"

    assert (
        conv_shape[0] == n_samples and conv_shape[1] == n_views
    ), f"ConvNeXt shape mismatch. Expected ({n_samples}, {n_views}, D), got {conv_shape}"

    assert (
        tab_shape[0] == n_samples and tab_shape[1] == 192
    ), f"Tabular shape mismatch. Expected ({n_samples}, 192), got {tab_shape}"

    print("  Feature Extraction validation passed.")

    # ==========================================
    # 4. Demonstrate Topology Transformation
    # ==========================================
    print("\n[Demo] Testing TopologyTransformer...")
    topology = TopologyTransformer()

    # 4a. Densification (Training Mode)
    # Should produce 9 centroids per sample
    print("  Applying Hyper-Densification (Training)...")
    densified = topology.densify_training_data(
        train_data["dino"],
        train_data["convnext"],
        train_data["tabular"],
        train_data["labels"],
        train_data["ids"],
        load_cached_data=False,  # Force computation
        cache_prefix="demo_train_densified",
    )

    expected_samples = n_samples * 9
    assert (
        densified["dino"].shape[0] == expected_samples
    ), f"Densification failed. Expected {expected_samples} samples, got {densified['dino'].shape[0]}"

    # Check that visual features are flattened from (N, Views, D) to (N*9, D)
    assert (
        len(densified["dino"].shape) == 2
    ), "Densified visual features should be 2D arrays."

    # 4b. Canonicalization (Inference Mode)
    # Should produce 1 centroid per sample
    print("  Applying Canonical Centroid generation (Inference)...")
    canonical = topology.create_inference_data(
        train_data["dino"],
        train_data["convnext"],
        train_data["tabular"],
        train_data["ids"],
        labels=train_data["labels"],
        load_cached_data=False,
        cache_prefix="demo_inference_canonical",
    )

    assert (
        canonical["dino"].shape[0] == n_samples
    ), f"Canonicalization failed. Expected {n_samples} samples, got {canonical['dino'].shape[0]}"
    assert (
        len(canonical["dino"].shape) == 2
    ), "Canonical visual features should be 2D arrays."

    print("  Topology Transformation validation passed.")

    # ==========================================
    # 5. Demonstrate Model Pipeline
    # ==========================================
    print("\n[Demo] Testing ModelFactory & Pipeline Training...")

    dino_dim = densified["dino"].shape[1]
    conv_dim = densified["convnext"].shape[1]
    tab_dim = densified["tabular"].shape[1]

    # Create Pipeline
    pipeline = ModelFactory.create_pipeline(dino_dim, conv_dim, tab_dim)
    assert isinstance(
        pipeline, Pipeline
    ), "Factory did not return a Scikit-Learn Pipeline."

    # Prepare Training Data (Concatenate features)
    X_train = np.hstack(
        [densified["dino"], densified["convnext"], densified["tabular"]]
    )
    y_train = densified["labels"]

    print(f"  Training Input Shape: {X_train.shape}")
    print(f"  Training Labels Shape: {y_train.shape}")

    # Fit Pipeline
    # Note: With only 4 samples * 9 = 36 rows, and many features,
    # LDA might complain about collinearity or classes, but it should run.
    # We ensure we have enough classes in the subset or handle the warning implicitly.
    # Since we picked the first 4 rows of train.csv, they might be the same class or different.
    # If there is only 1 class, LDA will fail. Let's check classes.
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        print(
            "  [Warning] Not enough classes in subset for LDA. Mocking labels for demo purposes."
        )
        # Create fake labels 0, 1, 0, 1...
        y_train = np.array([f"class_{i%2}" for i in range(len(y_train))])

    pipeline.fit(X_train, y_train)
    print("  Pipeline fitted successfully.")

    # Predict (Inference)
    X_test = np.hstack([canonical["dino"], canonical["convnext"], canonical["tabular"]])

    probs = pipeline.predict_proba(X_test)
    print(f"  Prediction Probabilities Shape: {probs.shape}")

    assert probs.shape[0] == n_samples, "Prediction row count mismatch."
    assert probs.shape[1] == len(pipeline.classes_), "Prediction column count mismatch."

    print("  Model Pipeline validation passed.")

    # ==========================================
    # 6. Demonstrate Submission Saving
    # ==========================================
    print("\n[Demo] Testing Submission Generation...")

    demo_sub_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    class_names = list(pipeline.classes_)
    ids = canonical["ids"]

    save_submission(ids, probs, class_names, output_path=demo_sub_path)

    # Verify file
    assert os.path.exists(demo_sub_path), "Submission file was not created."

    df_sub = pd.read_csv(demo_sub_path)
    print(f"  Saved submission shape: {df_sub.shape}")
    assert df_sub.shape == (
        n_samples,
        len(class_names) + 1,
    ), "Submission CSV dimensions incorrect (Classes + ID)."
    assert "id" in df_sub.columns, "ID column missing in submission."

    print("  Submission Generation validation passed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
