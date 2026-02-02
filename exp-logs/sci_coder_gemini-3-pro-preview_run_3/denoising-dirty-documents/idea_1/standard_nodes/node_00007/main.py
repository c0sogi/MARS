import os
import sys
import numpy as np
import pandas as pd
import warnings

# Add the current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library import config
import importlib

# Reload config to ensure updates are picked up in persistent sessions
# Cite debug_lesson_1: Restart the interpreter or force-reload modules when patching dependencies.
importlib.reload(config)
from library import utils
from library import data_loader
from library import model


def main():
    # --- 1. Setup ---
    print("--- Starting Runfile ---")
    np.random.seed(config.SEED)
    warnings.filterwarnings("ignore")

    # --- 2. Training ---
    print("\n--- Step 1: Training ---")
    # Extract training patches (uses caching if available)
    X_train, y_train = data_loader.extract_patch_data(
        metadata_path=config.TRAIN_METADATA_PATH,
        patch_size=config.PATCH_SIZE,
        num_samples=config.NUM_SAMPLES,
        load_cached_data=False,
    )

    # Initialize and train the model
    # Cite solution_lesson_node_00001: Replacing linear filter with CNN for better edge preservation.
    filter_model = model.DenoisingModel(patch_size=config.PATCH_SIZE)
    filter_model.fit(X_train, y_train)

    # --- 3. Validation ---
    print("\n--- Step 2: Validation ---")
    if not os.path.exists(config.VAL_METADATA_PATH):
        raise FileNotFoundError(
            f"Validation metadata not found at {config.VAL_METADATA_PATH}"
        )

    df_val = pd.read_csv(config.VAL_METADATA_PATH)

    val_sq_error_sum = 0.0
    val_pixel_count = 0

    # accumulators for failure analysis
    # We will store flattened arrays of input intensities and absolute errors
    fa_inputs = []
    fa_errors = []

    print(f"Validating on {len(df_val)} images...")

    for idx, row in df_val.iterrows():
        input_rel_path = row["input_path"]
        target_rel_path = row["target_path"]

        input_full_path = os.path.join(config.INPUT_DIR, input_rel_path)
        target_full_path = os.path.join(config.INPUT_DIR, target_rel_path)

        # Load images
        try:
            img_in = utils.load_normalized_image(input_full_path)
            img_tar = utils.load_normalized_image(target_full_path)
        except Exception as e:
            print(f"Error loading validation pair {input_rel_path}: {e}")
            continue

        # Inference
        # The provided model.predict uses cv2.filter2D (CPU optimized)
        img_pred = filter_model.predict(img_in)

        # Error Calculation
        diff = img_tar - img_pred
        val_sq_error_sum += np.sum(diff**2)
        val_pixel_count += diff.size

        # Collect data for failure analysis
        # Flatten and append to list
        fa_inputs.append(img_in.flatten())
        fa_errors.append(np.abs(diff).flatten())

    # Compute Global RMSE
    if val_pixel_count > 0:
        mse = val_sq_error_sum / val_pixel_count
        rmse = np.sqrt(mse)
    else:
        rmse = 0.0

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {rmse}")

    # --- 4. Failure Analysis ---
    print("\n--- Step 3: Failure Analysis ---")
    if len(fa_inputs) > 0:
        # Concatenate all pixels
        all_inputs = np.concatenate(fa_inputs)
        all_errors = np.concatenate(fa_errors)

        # Calculate Pearson Correlation
        # Using numpy corrcoef which returns a matrix [[1, r], [r, 1]]
        # We want the correlation between input intensity and error magnitude
        correlation_matrix = np.corrcoef(all_inputs, all_errors)
        correlation = correlation_matrix[0, 1]

        print(
            f"Correlation between Input Intensity and Absolute Error: {correlation:.4f}"
        )

        if abs(correlation) < 0.1:
            print("Interpretation: Error is roughly independent of pixel intensity.")
        elif correlation > 0:
            print(
                "Interpretation: Brighter pixels (background) tend to have higher errors."
            )
        else:
            print("Interpretation: Darker pixels (text) tend to have higher errors.")
    else:
        print("No validation data available for failure analysis.")

    # --- 5. Test Inference & Submission ---
    print("\n--- Step 4: Submission Generation ---")
    if not os.path.exists(config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {config.TEST_METADATA_PATH}"
        )

    df_test = pd.read_csv(config.TEST_METADATA_PATH)
    predictions_dict = {}

    print(f"Processing {len(df_test)} test images...")
    for idx, row in df_test.iterrows():
        input_rel_path = row["input_path"]
        image_id = row["image_id"]  # Filename e.g. '110.png'

        input_full_path = os.path.join(config.INPUT_DIR, input_rel_path)

        try:
            img_in = utils.load_normalized_image(input_full_path)
            # Predict
            img_pred = filter_model.predict(img_in)

            # Store prediction
            predictions_dict[image_id] = img_pred

        except Exception as e:
            print(f"Error processing test image {image_id}: {e}")
            # Fallback: if prediction fails, use input as prediction (no-op denoising)
            # This ensures submission generation doesn't crash
            try:
                predictions_dict[image_id] = utils.load_normalized_image(
                    input_full_path
                )
            except:
                pass

    # Format and save submission
    utils.format_submission(predictions_dict, config.SUBMISSION_PATH)
    print("Runfile execution complete.")


if __name__ == "__main__":
    main()
