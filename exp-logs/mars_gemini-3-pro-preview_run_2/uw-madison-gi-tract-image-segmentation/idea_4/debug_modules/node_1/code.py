import os
import sys
import numpy as np
import pandas as pd
import cv2
import joblib
import shutil
from scipy import ndimage

# Import provided library modules
from library import config, utils, data_processing, model, post_processing


def main():
    print("Starting End-to-End Pipeline Demonstration...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override config parameters for speed
    config.IMG_SIZE = (128, 128)  # Smaller images for faster processing
    config.N_SEGMENTS = 200  # Fewer superpixels
    config.N_ESTIMATORS = 10  # Minimal boosting rounds
    config.EARLY_STOPPING_ROUNDS = 5
    config.WORKING_DIR = "./working/demo_execution"
    config.SUBMISSION_PATH = os.path.join(config.WORKING_DIR, "submission.csv")

    # Ensure working directory exists
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    np.random.seed(config.SEED)

    print(
        f"Configuration optimized for demo: IMG_SIZE={config.IMG_SIZE}, N_ESTIMATORS={config.N_ESTIMATORS}"
    )

    # ==========================================
    # 2. Data Processing (Training Subset)
    # ==========================================
    print("\n--- Step 2: Data Processing ---")

    # Load training metadata
    try:
        train_meta = utils.load_metadata("train")
    except FileNotFoundError:
        print("Metadata not found. Ensure ./metadata/train_metadata.csv exists.")
        return

    # Select a single case for demonstration to keep it fast
    # We pick the first available case
    demo_case = train_meta["case"].unique()[0]
    demo_day = train_meta[train_meta["case"] == demo_case]["day"].unique()[0]

    print(f"Processing subset: Case {demo_case}, Day {demo_day}")

    subset_df = train_meta[
        (train_meta["case"] == demo_case) & (train_meta["day"] == demo_day)
    ].copy()

    # Process this group using the library function
    # _process_group expects ((case, day), dataframe)
    print("Generating superpixel features...")
    train_features = data_processing._process_group(
        ((demo_case, demo_day), subset_df), split="train"
    )

    # Validation
    assert not train_features.empty, "Feature extraction returned empty DataFrame."
    assert "label" in train_features.columns, "Labels missing from training features."
    print(
        f"Generated {len(train_features)} superpixel samples with {train_features.shape[1]} features."
    )

    # Create a dummy validation set (just a subset of train for this demo)
    val_features = train_features.sample(frac=0.2, random_state=config.SEED)
    train_features = train_features.drop(val_features.index)

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("\n--- Step 3: Model Training ---")

    # Instantiate and train
    clf = model.SuperpixelClassifier()
    clf.fit(train_features, val_features)

    # Validation
    assert clf.model is not None, "Model failed to initialize."
    print("Model training completed successfully.")

    # Save model
    model_path = os.path.join(config.WORKING_DIR, "demo_model.pth")
    clf.save(model_path)
    assert os.path.exists(model_path), "Model file was not saved."
    print(f"Model saved to {model_path}")

    # ==========================================
    # 4. Inference (Test Subset)
    # ==========================================
    print("\n--- Step 4: Inference ---")

    # Load test metadata
    test_meta = utils.load_metadata("test")

    # Select a single case for inference
    if not test_meta.empty:
        test_case = test_meta["case"].unique()[0]
        test_day = test_meta[test_meta["case"] == test_case]["day"].unique()[0]

        print(f"Running inference on Test Case {test_case}, Day {test_day}")
        test_subset = test_meta[
            (test_meta["case"] == test_case) & (test_meta["day"] == test_day)
        ].copy()

        # Run volume inference
        predictions = model._process_volume_inference(test_subset, clf)

        # Validate output format
        if predictions:
            print(f"Generated {len(predictions)} prediction entries.")
            first_pred = predictions[0]
            assert "id" in first_pred, "Prediction missing 'id'"
            assert "class" in first_pred, "Prediction missing 'class'"
            assert "predicted" in first_pred, "Prediction missing 'predicted' RLE"

            # Check RLE format (string of space-separated numbers)
            rle = first_pred["predicted"]
            if rle:
                parts = rle.split()
                assert (
                    len(parts) % 2 == 0
                ), "RLE string must have even number of elements"
                assert all(x.isdigit() for x in parts), "RLE must contain only digits"
        else:
            print(
                "Warning: No predictions generated (possibly empty images or filtering)."
            )
    else:
        print("Test metadata is empty, skipping inference step.")

    # ==========================================
    # 5. Post-Processing & Utility Verification
    # ==========================================
    print("\n--- Step 5: Logic Verification ---")

    # A. Verify RLE Encoding/Decoding
    print("Verifying RLE logic...")
    original_mask = np.zeros((10, 10), dtype=np.uint8)
    original_mask[2:5, 2:5] = 1  # 3x3 square

    rle_str = utils.rle_encode(original_mask)
    decoded_mask = utils.rle_decode(rle_str, (10, 10))

    assert np.array_equal(original_mask, decoded_mask), "RLE Decode mismatch!"
    print("RLE Encoding/Decoding passed.")

    # B. Verify 3D Post-Processing
    print("Verifying 3D cleaning logic...")
    # Create a 3D volume (Depth=3, H=10, W=10)
    vol = np.zeros((3, 10, 10), dtype=np.uint8)

    # Object 1 (Large): Present in all 3 slices
    vol[:, 1:4, 1:4] = 1

    # Object 2 (Small Noise): Present only in slice 0, far away
    vol[0, 8, 8] = 1

    cleaned_vol = post_processing.clean_3d_volume(vol)

    # The noise at [0, 8, 8] should be removed
    assert cleaned_vol[0, 8, 8] == 0, "Noise was not removed by 3D cleaning."
    # The main object should remain
    assert cleaned_vol[0, 2, 2] == 1, "Main object was incorrectly removed."
    print("3D Post-processing passed.")

    # ==========================================
    # 6. Generate Submission File
    # ==========================================
    print("\n--- Step 6: Submission Generation ---")

    # Create a dummy submission file based on our inference results
    if "predictions" in locals() and predictions:
        sub_df = pd.DataFrame(predictions)
        sub_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission file generated at {config.SUBMISSION_PATH}")
        assert os.path.exists(config.SUBMISSION_PATH)

        # Verify content
        loaded_sub = pd.read_csv(config.SUBMISSION_PATH)
        assert list(loaded_sub.columns) == [
            "id",
            "class",
            "predicted",
        ], "Submission columns incorrect"
        print(f"Submission has {len(loaded_sub)} rows.")

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
