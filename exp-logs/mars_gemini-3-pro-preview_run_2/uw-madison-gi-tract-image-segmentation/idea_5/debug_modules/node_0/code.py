import os
import numpy as np
import pandas as pd
import cv2
import random
import shutil

# Import from provided library files
from library.config import Config
from library.utils import (
    rle_encode,
    rle_decode,
    dice_coefficient,
    hausdorff_distance_3d,
    keep_largest_connected_component_3d,
)
from library.data_processing import (
    load_and_preprocess_image,
    generate_search_vector,
    set_seed,
)
from library.retrieval_system import AtlasSegmenter


def run_demo():
    print("=== Starting Demonstration of Retrieval-based Segmentation Library ===\n")

    # Ensure reproducibility
    set_seed(Config.SEED)
    random.seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Demonstrate Utility Functions
    # -------------------------------------------------------------------------
    print("--- 1. Testing Utility Functions ---")

    # A. RLE Encoding/Decoding
    print("Testing RLE Encoding/Decoding...")
    shape = (100, 100)
    original_mask = np.zeros(shape, dtype=np.uint8)
    # Create a simple square
    original_mask[20:40, 20:40] = 1

    rle_str = rle_encode(original_mask)
    decoded_mask = rle_decode(rle_str, shape)

    assert np.array_equal(
        original_mask, decoded_mask
    ), "RLE Decode failed to match original."
    print("  [Pass] RLE encode/decode cycle verified.")

    # B. Metrics
    print("Testing Metrics...")
    # Dice: Perfect overlap
    dice_score = dice_coefficient(original_mask, original_mask)
    assert np.isclose(dice_score, 1.0), f"Dice should be 1.0, got {dice_score}"

    # Dice: No overlap
    empty_mask = np.zeros(shape, dtype=np.uint8)
    dice_score_empty = dice_coefficient(original_mask, empty_mask)
    assert np.isclose(
        dice_score_empty, 0.0
    ), f"Dice should be 0.0, got {dice_score_empty}"

    # Hausdorff 3D
    # Create pseudo-3D volumes (Depth, H, W)
    vol_true = np.zeros((5, 100, 100), dtype=np.uint8)
    vol_pred = np.zeros((5, 100, 100), dtype=np.uint8)

    # Add object at same location
    vol_true[2, 50, 50] = 1
    vol_pred[2, 50, 51] = 1  # Shifted by 1 pixel in width

    hd_dist = hausdorff_distance_3d(vol_true, vol_pred)
    # Normalized distance: 1 pixel shift in 100 width = 0.01
    # Note: The implementation normalizes by (1, 1/H, 1/W).
    # Distance is sqrt((0)^2 + (0)^2 + (1/100)^2) = 0.01
    assert np.isclose(
        hd_dist, 0.01, atol=1e-4
    ), f"Hausdorff distance mismatch. Got {hd_dist}"
    print(f"  [Pass] Metrics verified (Dice: {dice_score}, HD: {hd_dist:.4f}).")

    # C. Post-processing (Largest Connected Component)
    print("Testing Post-processing...")
    noisy_vol = np.zeros((10, 50, 50), dtype=np.uint8)
    # Large component
    noisy_vol[2:5, 10:30, 10:30] = 1
    # Small noise component
    noisy_vol[8, 40:42, 40:42] = 1

    cleaned_vol = keep_largest_connected_component_3d(noisy_vol)

    # Verify noise is gone
    assert cleaned_vol[8, 40, 40] == 0, "Noise component was not removed."
    # Verify large component remains
    assert cleaned_vol[3, 20, 20] == 1, "Main component was incorrectly removed."
    print("  [Pass] Largest connected component filtering verified.")

    # -------------------------------------------------------------------------
    # 2. Demonstrate Data Processing
    # -------------------------------------------------------------------------
    print("\n--- 2. Testing Data Processing ---")

    # Load metadata to get a valid file path
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_row = df_train.iloc[0]
    file_path = sample_row["file_path"]

    print(f"Loading sample image: {file_path}")

    # Test Image Loading
    # Target size from config is (256, 256) usually, let's use that
    target_h, target_w = Config.IMG_SIZE
    img = load_and_preprocess_image(file_path, target_size=(target_h, target_w))

    assert img.shape == (target_h, target_w), f"Image shape mismatch. Got {img.shape}"
    assert img.dtype == np.float32, "Image should be float32"
    assert 0.0 <= img.min() and img.max() <= 1.0, "Image normalization failed."
    print("  [Pass] Image loading and preprocessing verified.")

    # Test Search Vector Generation
    search_vec = generate_search_vector(img, Config.SEARCH_SIZE)
    expected_dim = Config.SEARCH_SIZE[0] * Config.SEARCH_SIZE[1]
    assert search_vec.shape == (
        expected_dim,
    ), f"Vector dim mismatch. Got {search_vec.shape}"
    print("  [Pass] Search vector generation verified.")

    # -------------------------------------------------------------------------
    # 3. Demonstrate Retrieval System (Atlas Segmenter)
    # -------------------------------------------------------------------------
    print("\n--- 3. Testing Atlas Segmenter System ---")

    # Clean up working directory for a fresh demo run to ensure we test the processing logic
    # Note: In a real run, we'd keep the cache. Here we want to verify 'fit' works.
    if os.path.exists(Config.WORKING_DIR):
        print("Cleaning previous cache for demonstration purposes...")
        shutil.rmtree(Config.WORKING_DIR)
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    segmenter = AtlasSegmenter(config=Config)

    print("Fitting Atlas Segmenter (Debug Mode)...")
    # Debug mode processes only the first 2 cases, making this fast
    segmenter.fit(load_cached_data=False, debug=True)

    assert segmenter.atlas_vectors is not None, "Atlas vectors not initialized."
    assert segmenter.atlas_masks is not None, "Atlas masks not initialized."
    print(f"  [Pass] Atlas fitted. Index size: {len(segmenter.atlas_indices)}")

    # Simulate Prediction
    print("Running prediction on sample slice...")

    # We use the image we loaded earlier.
    # We need a relative depth. Let's assume 0.5 (middle of organ)
    test_depth = 0.5

    # Predict
    pred_mask = segmenter.predict_slice(img, test_depth)

    # Validation
    expected_shape = (Config.IMG_SIZE[0], Config.IMG_SIZE[1], len(Config.CLASSES))
    assert (
        pred_mask.shape == expected_shape
    ), f"Prediction shape mismatch. Got {pred_mask.shape}"
    assert pred_mask.dtype == np.uint8, "Prediction should be uint8"
    assert np.isin(pred_mask, [0, 1]).all(), "Prediction should be binary."

    print(f"  [Pass] Prediction successful. Output shape: {pred_mask.shape}")

    # Check if we got any positive predictions (might be all zeros if no match found, which is valid)
    # Since we used an image from the dataset (likely one of the first ones),
    # and we fit on the first 2 cases, we should ideally find a match.
    pixel_sum = np.sum(pred_mask)
    print(f"  Total positive pixels predicted: {pixel_sum}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
