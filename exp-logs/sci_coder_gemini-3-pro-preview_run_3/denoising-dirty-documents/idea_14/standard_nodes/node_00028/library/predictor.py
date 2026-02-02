import os
import cv2
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.network import DnCNN
from library.utils import create_submission_file


def get_test_data():
    """
    Reads test metadata and loads images.
    Returns a list of dictionaries containing image ID and normalized image data.
    """
    if not os.path.exists(Config.TEST_CSV):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_CSV}")

    df = pd.read_csv(Config.TEST_CSV)
    test_data = []

    # Load images
    for _, row in df.iterrows():
        img_id = row["image_id"]
        rel_path = row["input_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load as grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load test image {full_path}")
            continue

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        test_data.append({"id": img_id, "image": img})

    return test_data


def apply_tta(model, x_tensor):
    """
    Applies 8 geometric transformations (D4 group), predicts noise,
    inverses transformations, and averages the results.

    Args:
        model: The neural network model.
        x_tensor: Input tensor of shape [1, 1, H, W].

    Returns:
        torch.Tensor: Averaged noise prediction of shape [1, 1, H, W].
    """
    accumulated_noise = torch.zeros_like(x_tensor)

    # Define transformations as (k_rotations, horizontal_flip)
    # k=0: 0 deg, k=1: 90 deg CCW, k=2: 180 deg, k=3: 270 deg CCW
    transforms = [
        (0, False),
        (1, False),
        (2, False),
        (3, False),
        (0, True),
        (1, True),
        (2, True),
        (3, True),
    ]

    for k, flip in transforms:
        # --- 1. Forward Transform ---
        x_aug = x_tensor.clone()

        # Apply Flip (Horizontal along Width axis, dim 3)
        if flip:
            x_aug = torch.flip(x_aug, dims=[3])

        # Apply Rotation (Plane H, W -> dims 2, 3)
        if k > 0:
            x_aug = torch.rot90(x_aug, k, dims=[2, 3])

        # --- 2. Inference ---
        with torch.no_grad():
            pred_noise_aug = model(x_aug)

        # --- 3. Inverse Transform ---
        # We must reverse the operations in LIFO order:
        # y = Rot(Flip(x))  ->  x = Flip_inv(Rot_inv(y))

        pred_noise = pred_noise_aug

        # Inverse Rotation: Rotate by -k (or 4-k)
        if k > 0:
            pred_noise = torch.rot90(pred_noise, -k, dims=[2, 3])

        # Inverse Flip: Flip again
        if flip:
            pred_noise = torch.flip(pred_noise, dims=[3])

        accumulated_noise += pred_noise

    # Average over the 8 transformations
    return accumulated_noise / len(transforms)


def inference_pipeline():
    """
    Main function to run the inference pipeline.
    Loads models, processes test images with TTA, ensembles results, and saves submission.
    """
    print("Initializing Inference Pipeline...")

    device = torch.device(Config.DEVICE)

    # 1. Load Test Data
    try:
        test_data = get_test_data()
        print(f"Loaded {len(test_data)} test images.")
    except Exception as e:
        print(f"Failed to load test data: {e}")
        return

    if not test_data:
        print("No test data found.")
        return

    # Initialize accumulation dictionary for ensemble averaging
    # Key: image_id, Value: Accumulated Noise Prediction (numpy array)
    ensemble_noise_sums = {
        item["id"]: np.zeros_like(item["image"]) for item in test_data
    }
    valid_models_count = 0

    # 2. Iterate over Ensemble Members
    for member_id in range(Config.ENSEMBLE_SIZE):
        model_filename = f"model_{member_id}.pth"
        model_path = os.path.join(Config.WORKING_DIR, model_filename)

        if not os.path.exists(model_path):
            print(
                f"Checkpoint for Member {member_id} not found at {model_path}. Skipping."
            )
            continue

        print(f"--- Processing with Member {member_id} ---")

        # Initialize Model Architecture
        model = DnCNN(
            in_channels=Config.IN_CHANNELS,
            out_channels=Config.OUT_CHANNELS,
            num_features=Config.NUM_FEATURES,
            num_blocks=Config.NUM_RES_BLOCKS,
        ).to(device)

        # Load Weights
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            print(f"Error loading model {member_id}: {e}")
            continue

        valid_models_count += 1

        # Process all images for this model
        for item in test_data:
            img_id = item["id"]
            img = item["image"]

            # Prepare input tensor: [1, 1, H, W]
            x_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

            # Get TTA-averaged prediction
            noise_pred_tensor = apply_tta(model, x_tensor)

            # Convert back to numpy
            noise_pred = noise_pred_tensor.squeeze().cpu().numpy()

            # Accumulate
            ensemble_noise_sums[img_id] += noise_pred

    # 3. Finalize Predictions
    if valid_models_count == 0:
        print("Error: No valid models were loaded. Aborting submission generation.")
        return

    print(f"Aggregating predictions from {valid_models_count} models...")

    final_predictions = {}
    for item in test_data:
        img_id = item["id"]
        noisy_input = item["image"]

        # Average the accumulated noise predictions
        avg_noise_pred = ensemble_noise_sums[img_id] / valid_models_count

        # Reconstruct Clean Image: Clean = Noisy - Predicted_Noise
        clean_pred = noisy_input - avg_noise_pred

        # Clip values to valid range [0, 1]
        clean_pred = np.clip(clean_pred, 0.0, 1.0)

        final_predictions[img_id] = clean_pred

    # 4. Generate Submission File
    print(f"Generating submission file at {Config.SUBMISSION_FILE}...")
    create_submission_file(final_predictions, Config.SUBMISSION_FILE)
    print("Inference pipeline completed successfully.")
