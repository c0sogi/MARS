import os
import shutil
import warnings
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything, clip_probabilities
from library.image_processing import LeafImageProcessor
from library.data_manager import DataManager
from library.model_pipeline import LDAPipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Leaf Classification Pipeline Demo ===")

    # 1. Configuration & Setup
    # ------------------------
    seed_everything(42)

    # Define a temporary working directory for this demo
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths to use the demo directory
    # This ensures we don't overwrite any actual experiment data
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    print(f"Working directory set to: {Config.WORKING_DIR}")

    # 2. Prepare Data Subset
    # ----------------------
    # We need a small subset of data that contains at least 2 classes
    # to demonstrate the LDA classifier successfully.
    print("\n[Step 1] Preparing Data Subset...")
    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Select top 2 classes
    classes = full_train_df["species"].unique()[:2]
    # Take 2 images from each of these 2 classes (Total 4 images)
    subset_df = (
        full_train_df[full_train_df["species"].isin(classes)]
        .groupby("species")
        .head(2)
        .reset_index(drop=True)
    )

    print(f"Selected subset: {len(subset_df)} images across {len(classes)} classes.")
    print(subset_df[["id", "species", "file_path"]])

    # 3. Instantiate DataManager
    # --------------------------
    # This initializes the ImageProcessor and DualStreamExtractor (loads DINOv2 + ConvNeXt)
    print("\n[Step 2] Initializing DataManager (Loading Models)...")
    dm = DataManager()

    # 4. Verify Image Processing Logic
    # ------------------------------
    print("\n[Step 3] Verifying Image Processing...")
    processor = dm.img_processor
    sample_path = subset_df.iloc[0]["file_path"]

    # Load Image
    img = processor.load_image(sample_path)
    assert isinstance(img, np.ndarray), "Loaded image should be a numpy array"
    assert img.shape[2] == 3, "Image should be RGB (3 channels)"

    # Generate Views
    views = processor.generate_rotated_views(img)
    assert len(views) == 12, "Should generate 12 views"
    assert views[0].shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"View shape mismatch. Expected (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {views[0].shape}"

    print("Image processing logic verified: 12 views generated correctly.")

    # 5. Verify Feature Extraction Logic
    # --------------------------------
    print("\n[Step 4] Verifying Feature Extraction...")
    # We use the extractor loaded within DataManager
    extractor = dm.feature_extractor

    # Create a batch from the views of the single image
    batch_tensor = torch.stack(views)

    # Extract
    features = extractor.extract_batch(batch_tensor)

    # Expected Dimensions: DINOv2-Large (1024) + ConvNeXt-Large (1536) = 2560
    expected_dim = 1024 + 1536
    assert features.shape == (
        12,
        expected_dim,
    ), f"Feature shape mismatch. Expected (12, {expected_dim}), got {features.shape}"

    print(f"Feature extraction logic verified. Output vector shape: {features.shape}")

    # 6. Execute Data Pipeline on Subset
    # ----------------------------------
    print("\n[Step 5] Running Data Pipeline on Subset...")

    # A. Extract Features for all images in subset
    # Note: This saves to disk in demo_dir
    subset_img_features = dm.extract_all_views(
        subset_df, "demo_subset", load_cached_data=False
    )
    assert subset_img_features.shape == (4, 12, expected_dim)

    # B. Process Tabular Features
    # We use the same subset for train/val/test just to verify the function works
    train_tab, val_tab, test_tab = dm.process_tabular_features(
        subset_df, subset_df, subset_df
    )
    # 3 feature groups * 64 attributes = 192 features
    assert train_tab.shape == (4, 192)

    # C. Manifold Densification (Training Topology)
    # Should result in 3 centroids per image -> 4 * 3 = 12 samples
    labels = subset_df["species"].values
    dense_img, dense_tab, dense_y = dm.densify_training_data(
        subset_img_features, train_tab, labels
    )

    assert dense_img.shape == (12, expected_dim)
    assert dense_tab.shape == (12, 192)
    assert dense_y.shape == (12,)
    print(
        f"Densification verified. Expanded {len(subset_df)} samples to {len(dense_y)} samples."
    )

    # D. Feature Fusion
    X_train = dm.fuse_features(dense_img, dense_tab)
    assert X_train.shape == (12, expected_dim + 192)

    # E. Inference Preparation (Validation/Test Topology)
    # Should result in 1 centroid per image -> 4 samples
    inf_img, inf_tab = dm.prepare_inference_data(subset_img_features, train_tab)
    X_inference = dm.fuse_features(inf_img, inf_tab)

    assert X_inference.shape == (4, expected_dim + 192)
    print("Inference preparation verified.")

    # 7. Verify Model Pipeline
    # ------------------------
    print("\n[Step 6] Verifying Model Pipeline (LDA)...")
    pipeline = LDAPipeline()

    # Fit on densified data
    pipeline.fit(X_train, dense_y)
    print("Model fitted successfully.")

    # Predict on inference data
    probs = pipeline.predict(X_inference)

    # Verify predictions
    assert probs.shape == (
        4,
        2,
    ), f"Prediction shape mismatch. Expected (4, 2), got {probs.shape}"

    # Verify clipping utility
    clipped_probs = clip_probabilities(probs)
    assert clipped_probs.min() >= Config.PROB_CLIP_MIN
    assert clipped_probs.max() <= Config.PROB_CLIP_MAX

    print("Prediction verified. Sample probabilities:\n", clipped_probs)

    # 8. Generate Submission File
    # ---------------------------
    print("\n[Step 7] Generating Demo Submission...")
    submission_df = pd.DataFrame(clipped_probs, columns=pipeline.classes_)
    submission_df.insert(0, "id", subset_df["id"].values)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    assert os.path.exists(Config.SUBMISSION_PATH)
    print(f"Submission file created at {Config.SUBMISSION_PATH}")
    print(submission_df.head())

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
