import sys
import os
import numpy as np
import pandas as pd
import torch

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, load_image, rotate_image
from library.feature_extraction import (
    DualStreamExtractor,
    extract_dataset_features,
    get_or_compute_features,
)
from library.data_processing import (
    create_orthogonal_centroids,
    get_processed_data,
    get_stratified_folds,
)
from library.modeling import (
    SelectiveTopologyPipeline,
    train_ensemble,
    generate_submission,
)


def main():
    print(
        "Starting demonstration of the Selective-Topology Orthogonal Manifold-Densified LDA solution..."
    )

    # 1. Setup and Reproducibility
    seed_everything(42)

    # Define a small sample limit for speed
    DEMO_SAMPLE_LIMIT = 6
    Config.DEBUG_SAMPLE_LIMIT = DEMO_SAMPLE_LIMIT
    # Reduce folds for demo purposes to avoid class count issues with small samples
    Config.N_FOLDS = 2

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load metadata to find a valid image path
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_row = train_meta.iloc[0]
    sample_img_path = os.path.join(Config.INPUT_DIR, sample_row["file_path"])

    print("\n=== 1. Demonstrating library.utils ===")

    # Test load_image
    print(f"Loading image from: {sample_img_path}")
    img = load_image(sample_img_path)
    print(f"Image loaded. Shape: {img.shape}, Type: {img.dtype}")

    assert isinstance(img, np.ndarray), "Loaded image is not a numpy array"
    assert img.ndim == 3 and img.shape[2] == 3, "Image should be RGB (H, W, 3)"
    assert img.dtype == np.uint8, "Image dtype should be uint8"

    # Test rotate_image
    angle = 45
    print(f"Rotating image by {angle} degrees...")
    rot_img = rotate_image(img, angle)

    assert rot_img.shape == img.shape, "Rotated image shape mismatch"
    # Check if background padding (white) is applied (corner pixels likely white in rotated image)
    print("Rotation successful.")

    print("\n=== 2. Demonstrating library.feature_extraction ===")

    # Initialize Extractor
    print("Initializing DualStreamExtractor (DINOv2 + ConvNeXt)...")
    extractor = DualStreamExtractor()

    # Extract features for a small subset manually
    # Fix: Select 2 images from top 2 species to satisfy LDA (min 2 classes) and StratifiedKFold (min 2 samples/class)
    # Cite debug_lesson_12
    top_species = train_meta["species"].value_counts().head(2).index
    subset_meta = (
        train_meta[train_meta["species"].isin(top_species)]
        .groupby("species")
        .head(2)
        .reset_index(drop=True)
    )
    print(f"Extracting features for {len(subset_meta)} images (12 rotations each)...")

    df_features = extract_dataset_features(subset_meta, extractor)

    print(f"Extracted features shape: {df_features.shape}")

    # Assertions
    expected_rows = len(subset_meta) * Config.NUM_ROTATIONS  # 2 * 12 = 24
    assert (
        len(df_features) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_features)}"
    assert "feat_0" in df_features.columns, "Feature columns missing"
    assert "view_idx" in df_features.columns, "view_idx column missing"

    # Check feature dimension (DINO ~1024 + ConvNeXt ~1536 = ~2560)
    n_features = len([c for c in df_features.columns if c.startswith("feat_")])
    print(f"Feature vector dimension: {n_features}")
    assert n_features > 1000, "Feature dimension seems too low"

    print("\n=== 3. Demonstrating library.data_processing ===")

    # Test Densification (Orthogonal Centroids)
    print("Creating orthogonal centroids (Densification)...")
    df_densified = create_orthogonal_centroids(df_features)

    print(f"Densified data shape: {df_densified.shape}")

    # Assertions
    # 2 images * 3 centroids = 6 rows
    expected_densified_rows = len(subset_meta) * 3
    assert (
        len(df_densified) == expected_densified_rows
    ), f"Expected {expected_densified_rows} rows, got {len(df_densified)}"
    assert "centroid_type" in df_densified.columns, "centroid_type column missing"

    unique_centroids = sorted(df_densified["centroid_type"].unique())
    assert unique_centroids == [
        "A",
        "B",
        "C",
    ], f"Unexpected centroids: {unique_centroids}"

    # Test Stratified Folds
    print("Generating Stratified Folds...")
    # We need enough samples for folds. The demo subset of 2 is too small for 10 folds usually.
    # Let's use the df_densified we just made but with n_folds=2 for demonstration
    folds = get_stratified_folds(df_densified, n_folds=2, seed=42)

    print(f"Generated {len(folds)} folds.")
    for i, (train_idx, val_idx) in enumerate(folds):
        print(f"  Fold {i}: Train Size={len(train_idx)}, Val Size={len(val_idx)}")
        # Verify indices are within bounds
        assert train_idx.max() < len(df_densified)
        assert val_idx.max() < len(df_densified)
        # Verify no overlap
        assert (
            len(np.intersect1d(train_idx, val_idx)) == 0
        ), "Train and Val indices overlap"

    print("\n=== 4. Demonstrating library.modeling (Pipeline) ===")

    # Instantiate Pipeline
    pipeline = SelectiveTopologyPipeline()

    # Prepare data
    y = df_densified["species"].values

    # Fit
    print("Fitting SelectiveTopologyPipeline...")
    pipeline.fit(df_densified, y)

    # Predict
    print("Predicting probabilities...")
    probs = pipeline.predict_proba(df_densified)

    print(f"Probabilities shape: {probs.shape}")
    # Cite debug_lesson_2: The model output shape corresponds to the observed classes in the subset, not the global label space.
    n_classes_subset = len(np.unique(y))
    assert probs.shape == (
        len(df_densified),
        n_classes_subset,
    ), f"Probability shape mismatch (should be N_samples x {n_classes_subset} classes)"

    # Check probability properties
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # Save/Load check
    model_path = os.path.join(Config.WORKING_DIR, "demo_model.pkl")
    pipeline.save(model_path)
    loaded_pipeline = SelectiveTopologyPipeline.load(model_path)
    print("Model serialization verified.")

    print("\n=== 5. Demonstrating Full Training Ensemble ===")

    # Run train_ensemble with a small limit
    # This will trigger get_processed_data -> get_or_compute_features -> extraction -> caching
    # It will then run the K-Fold training
    print(f"Running train_ensemble with sample_limit={DEMO_SAMPLE_LIMIT}...")

    try:
        train_ensemble(sample_limit=DEMO_SAMPLE_LIMIT)
    except Exception as e:
        print(f"Ensemble training failed: {e}")
        raise e

    # Check if models were saved
    model_files = os.listdir(Config.CACHE_PATH_MODELS)
    print(f"Models saved in {Config.CACHE_PATH_MODELS}: {len(model_files)}")
    assert len(model_files) > 0, "No models were saved during ensemble training"

    print("\n=== 6. Demonstrating Submission Generation ===")

    # Run generate_submission with a small limit
    print(f"Running generate_submission with sample_limit={DEMO_SAMPLE_LIMIT}...")

    try:
        generate_submission(sample_limit=DEMO_SAMPLE_LIMIT)
    except Exception as e:
        print(f"Submission generation failed: {e}")
        raise e

    # Check submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated. Shape: {sub_df.shape}")
        print(sub_df.head(2))
        assert (
            sub_df.shape[1] == 100
        ), "Submission should have 100 columns (id + 99 species)"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
