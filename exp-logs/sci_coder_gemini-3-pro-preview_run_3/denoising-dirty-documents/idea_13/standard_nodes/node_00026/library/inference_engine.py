import os
import cv2
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.model import EZ_ResDnCNN
from library.utils import apply_tta, reverse_tta


def load_ensemble(device, ensemble_size=None):
    """
    Loads the ensemble of trained models.

    Args:
        device (torch.device): The device to load models onto.
        ensemble_size (int): Number of models in the ensemble.

    Returns:
        list: A list of loaded PyTorch models set to eval mode.
    """
    if ensemble_size is None:
        ensemble_size = Config.ENSEMBLE_SIZE

    models = []
    print(f"Loading ensemble of {ensemble_size} models...")

    for i in range(ensemble_size):
        # Construct model path assuming standard naming convention from training
        # e.g., "model_0.pth", "model_1.pth" inside WORKING_DIR
        model_path = os.path.join(Config.WORKING_DIR, f"model_{i}.pth")

        if not os.path.exists(model_path):
            print(f"Warning: Model checkpoint {model_path} not found. Skipping.")
            continue

        model = EZ_ResDnCNN()
        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        models.append(model)

    print(f"Successfully loaded {len(models)} models.")
    return models


def predict_image(image_path, models, device, tta_steps=None):
    """
    Performs inference on a single image using the model ensemble and TTA.

    Args:
        image_path (str): Path to the noisy image.
        models (list): List of loaded PyTorch models.
        device (torch.device): Computation device.
        tta_steps (int): Number of TTA transformations (up to 8).

    Returns:
        np.ndarray: The denoised image (H, W) with values in [0, 1].
    """
    if tta_steps is None:
        tta_steps = Config.TTA_STEPS

    # 1. Load and Preprocess
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Convert to Tensor (1, 1, H, W)
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    # 2. Ensemble Inference with TTA
    total_noise_pred = torch.zeros_like(img_tensor)
    count = 0

    with torch.no_grad():
        for model in models:
            for k in range(tta_steps):
                # Apply TTA
                aug_input = apply_tta(img_tensor, k)

                # Predict Noise
                aug_noise = model(aug_input)

                # Reverse TTA
                pred_noise = reverse_tta(aug_noise, k)

                total_noise_pred += pred_noise
                count += 1

    # 3. Average and Reconstruct
    avg_noise_pred = total_noise_pred / count
    clean_pred_tensor = img_tensor - avg_noise_pred

    # Clamp to valid range
    clean_pred_tensor = torch.clamp(clean_pred_tensor, 0.0, 1.0)

    # Convert back to numpy (H, W)
    clean_pred = clean_pred_tensor.squeeze().cpu().numpy()

    return clean_pred


def create_submission_file(output_path=None):
    """
    Generates the submission file for the test set.

    Args:
        output_path (str): Path to save the CSV file.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    device = torch.device(Config.DEVICE)

    # 1. Load Models
    models = load_ensemble(device)
    if not models:
        print("Error: No models loaded. Cannot generate submission.")
        return

    # 2. Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        print(f"Error: Test metadata not found at {Config.TEST_METADATA_PATH}")
        return

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    print(f"Generating predictions for {len(df_test)} test images...")

    # 3. Open Output File
    # We write line-by-line or chunk-by-chunk to handle large file size efficiently
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        # Write Header
        f.write("id,value\n")

        for idx, row in df_test.iterrows():
            image_id_full = row["image_id"]
            rel_path = row["input_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            # Extract ID without extension (e.g., "110.png" -> "110")
            image_id_base = os.path.splitext(image_id_full)[0]

            # Predict
            try:
                denoised_img = predict_image(full_path, models, device)
            except Exception as e:
                print(f"Error predicting {image_id_full}: {e}")
                continue

            # Flatten and Format
            h, w = denoised_img.shape

            # Vectorized generation of strings for speed
            # Create coordinate grids (1-based indexing)
            # rows: 1..H, cols: 1..W
            rows, cols = np.indices((h, w))
            rows = rows + 1
            cols = cols + 1

            # Flatten arrays
            flat_vals = denoised_img.flatten()
            flat_rows = rows.flatten()
            flat_cols = cols.flatten()

            # Create formatted lines
            # Format: {image_id}_{row}_{col},{value}
            # Using list comprehension or map is often faster than pandas for simple string formatting
            # However, for millions of pixels, we want to be careful.
            # Let's construct a large string buffer for the image.

            lines = [
                f"{image_id_base}_{r}_{c},{v:.6f}"
                for r, c, v in zip(flat_rows, flat_cols, flat_vals)
            ]

            # Write to file
            f.write("\n".join(lines))
            f.write("\n")

            if (idx + 1) % 5 == 0:
                print(f"Processed {idx + 1}/{len(df_test)} images")

    print(f"Submission saved to {output_path}")
