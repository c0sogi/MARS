import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import RDN
from library.utils import load_image, save_submission, seed_everything


def predict_full_image(model, image_path, device):
    """
    Performs inference on a single full-resolution image.

    Args:
        model (nn.Module): The trained RDN model.
        image_path (str): Path to the noisy image file.
        device (torch.device): Device to perform inference on.

    Returns:
        np.ndarray: The denoised image array.
    """
    # Load image (normalized [0, 1], grayscale)
    img_in = load_image(image_path)

    # Prepare input tensor: (H, W) -> (1, 1, H, W)
    img_tensor = torch.from_numpy(img_in).float().unsqueeze(0).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        # Model predicts the noise residual
        noise_pred = model(img_tensor)

    # Convert prediction back to numpy
    noise_pred_np = noise_pred.squeeze().cpu().numpy()

    # Reconstruct Clean Image: Input - Predicted Noise
    clean_pred = img_in - noise_pred_np

    return clean_pred


def generate_submission(
    model_path=Config.MODEL_PATH, submission_output=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model_path (str): Path to the trained model weights.
        submission_output (str): Path to save the generated submission CSV.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing inference on device: {device}")

    # Initialize Model
    model = RDN().to(device)

    # Load Weights
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {model_path}. Using untrained model (random weights)."
        )

    model.eval()

    # Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_METADATA}")

    df_test = pd.read_csv(Config.TEST_METADATA)
    print(f"Found {len(df_test)} test images.")

    predictions = {}

    # Iterate through test images
    for _, row in df_test.iterrows():
        img_id = row["image_id"]
        input_rel_path = row["input_path"]
        input_full_path = os.path.join(Config.INPUT_DIR, input_rel_path)

        try:
            clean_img = predict_full_image(model, input_full_path, device)
            predictions[img_id] = clean_img
        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            # Fallback: if inference fails, use the input image as the prediction
            # (better than crashing or empty submission)
            try:
                predictions[img_id] = load_image(input_full_path)
            except:
                pass

    # Save Submission
    print(f"Saving submission to {submission_output}...")
    save_submission(predictions, submission_path=submission_output)
    print("Submission generation completed.")
