import os
import cv2
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, normalize_image, denormalize_image
from library.model import DRDN


def generate_predictions():
    """
    Generates predictions for the test set using the trained DRDN model.
    Loads the model from Config.MODEL_SAVE_PATH, processes images listed in
    Config.TEST_METADATA_PATH, and saves the formatted submission to
    Config.SUBMISSION_PATH.
    """
    # 1. Setup Environment
    set_seed(Config.SEED)
    Config.create_directories()
    device = torch.device(Config.DEVICE)

    # 2. Load Model
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Error: Model checkpoint not found at {Config.MODEL_SAVE_PATH}")
        return

    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    model = DRDN().to(device)
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        print(f"Error: Test metadata not found at {Config.TEST_METADATA_PATH}")
        return

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    print(f"Found {len(df_test)} test images for inference.")

    # 4. Inference Loop
    submission_data = []

    with torch.no_grad():
        for _, row in df_test.iterrows():
            image_filename = row["image_id"]
            # Extract base ID (e.g., "110.png" -> "110")
            image_id_base = os.path.splitext(image_filename)[0]

            input_rel_path = row["input_path"]
            input_full_path = os.path.join(Config.INPUT_DIR, input_rel_path)

            # Load Image (Grayscale)
            img_in = cv2.imread(input_full_path, cv2.IMREAD_GRAYSCALE)
            if img_in is None:
                print(f"Warning: Could not load image {input_full_path}")
                continue

            h, w = img_in.shape

            # Normalize to [0, 1]
            img_norm = normalize_image(img_in)

            # Prepare Tensor: (1, 1, H, W)
            input_tensor = (
                torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
            )

            # Predict Noise
            # Model is fully convolutional, handling full resolution
            noise_pred = model(input_tensor)

            # Reconstruct Clean Image: Input - Noise
            clean_pred = input_tensor - noise_pred

            # Clamp to valid range [0, 1]
            clean_pred = torch.clamp(clean_pred, 0.0, 1.0)

            # Convert to Numpy and Denormalize to [0, 255] uint8
            # The metric compares against grayscale [0-255], so we submit integers.
            clean_pred_np = clean_pred.squeeze().cpu().numpy()
            clean_pred_uint8 = denormalize_image(clean_pred_np)

            # 5. Format for Submission
            # Flatten pixel values
            flat_vals = clean_pred_uint8.flatten()

            # Generate IDs: "{image_id}_{row}_{col}" (1-based indexing)
            # Use numpy to generate indices efficiently
            # Rows: [1, 1, ..., 2, 2, ...]
            r_indices = np.repeat(np.arange(1, h + 1), w)
            # Cols: [1, 2, ..., 1, 2, ...]
            c_indices = np.tile(np.arange(1, w + 1), h)

            # Create ID strings
            # List comprehension is reasonably fast for ~200k pixels per image
            current_ids = [
                f"{image_id_base}_{r}_{c}" for r, c in zip(r_indices, c_indices)
            ]

            # Append to main list
            submission_data.extend(zip(current_ids, flat_vals))

    # 6. Save Submission
    print(f"Generating submission file with {len(submission_data)} rows...")
    df_submission = pd.DataFrame(submission_data, columns=["id", "value"])

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
