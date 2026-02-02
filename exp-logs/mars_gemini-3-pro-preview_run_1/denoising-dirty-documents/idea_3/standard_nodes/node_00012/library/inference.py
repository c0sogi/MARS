import os
import torch
import numpy as np
from library.config import Config
from library.dataset import DenoisingDataset
from library.model import AttentionUNet
from library.utils import create_submission_csv


def predict_with_tta(model, x, device):
    """
    Performs inference on a single image using Test-Time Augmentation (TTA).
    Augmentations: Original, Horizontal Flip, Vertical Flip, Rotate 90.

    Args:
        model (torch.nn.Module): The trained model.
        x (torch.Tensor): Input image tensor of shape (1, 1, H, W).
        device (torch.device): The computation device.

    Returns:
        torch.Tensor: The averaged prediction tensor of shape (1, 1, H, W).
    """
    # Create augmented versions
    # x is (1, 1, H, W)

    # 1. Original
    x_orig = x

    # 2. Horizontal Flip (flip along width axis, dim 3)
    x_h = torch.flip(x, [3])

    # 3. Vertical Flip (flip along height axis, dim 2)
    x_v = torch.flip(x, [2])

    # 4. Rotate 90 degrees (counter-clockwise, dims 2 & 3)
    x_r = torch.rot90(x, 1, [2, 3])

    # Check dimensions to handle non-square images
    h, w = x.shape[2], x.shape[3]

    if h == w:
        # Concatenate into a single batch for efficient inference: (4, 1, H, W)
        batch = torch.cat([x_orig, x_h, x_v, x_r], dim=0).to(device)

        # Forward pass
        with torch.no_grad():
            logits = model(batch)
            preds = torch.sigmoid(logits)

        # Split predictions
        p_orig = preds[0:1]
        p_h = preds[1:2]
        p_v = preds[2:3]
        p_r = preds[3:4]
    else:
        # For non-square images, x_r has shape (1, 1, W, H) which cannot be concatenated
        # with x_orig (1, 1, H, W). Process x_r separately.
        batch_main = torch.cat([x_orig, x_h, x_v], dim=0).to(device)

        with torch.no_grad():
            # Inference on main batch
            logits_main = model(batch_main)
            preds_main = torch.sigmoid(logits_main)

            # Inference on rotated image
            logits_r = model(x_r)
            preds_r = torch.sigmoid(logits_r)

        # Split predictions
        p_orig = preds_main[0:1]
        p_h = preds_main[1:2]
        p_v = preds_main[2:3]
        p_r = preds_r

    # Inverse transformations to align predictions
    p_h_inv = torch.flip(p_h, [3])
    p_v_inv = torch.flip(p_v, [2])
    p_r_inv = torch.rot90(p_r, 3, [2, 3])  # Rotate 270 deg CCW (equivalent to -90)

    # Average the predictions
    avg_pred = (p_orig + p_h_inv + p_v_inv + p_r_inv) / 4.0

    return avg_pred


def generate_submission(debug=False):
    """
    Loads the ensemble of trained models, performs inference with TTA on the test set,
    aggregates predictions, and generates the submission CSV file.

    Args:
        debug (bool): If True, processes a subset of the test data for debugging.
    """
    print("Initializing submission generation...")

    device = torch.device(Config.DEVICE)

    # --- 1. Load Ensemble Models ---
    models = []
    for i in range(Config.NUM_MODELS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_{i}.pth")

        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Skipping.")
            continue

        try:
            # Initialize model architecture
            model = AttentionUNet(n_channels=1, n_classes=1).to(device)

            # Load weights
            checkpoint = torch.load(model_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            models.append(model)
            print(f"Loaded model {i} from {model_path}")

        except Exception as e:
            print(f"Error loading model {i}: {e}")

    if not models:
        raise RuntimeError("No valid models loaded. Cannot generate submission.")

    print(f"Ensemble size: {len(models)}")

    # --- 2. Load Test Data ---
    # We use the dataset directly to handle variable image sizes (batch_size=1)
    test_dataset = DenoisingDataset(split="test", debug=debug)
    print(f"Test set size: {len(test_dataset)} images")

    predictions = {}

    # --- 3. Inference Loop ---
    print("Starting inference...")

    for idx in range(len(test_dataset)):
        # Get data: noisy_t is (1, H, W), img_id is str
        noisy_t, img_id = test_dataset[idx]

        # Prepare input: (1, 1, H, W)
        input_tensor = noisy_t.unsqueeze(0).to(device)

        # Ensemble Aggregation
        ensemble_accum = None

        for model in models:
            # Get prediction with TTA for this model
            pred = predict_with_tta(model, input_tensor, device)

            if ensemble_accum is None:
                ensemble_accum = pred
            else:
                ensemble_accum += pred

        # Average across models
        final_pred = ensemble_accum / len(models)

        # Post-process: Squeeze to (H, W) and convert to numpy
        final_pred_np = final_pred.squeeze().cpu().numpy()

        # Store prediction
        predictions[img_id] = final_pred_np

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(test_dataset)} images")

    # --- 4. Generate CSV ---
    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    print(f"Generating submission file at {output_path}...")

    create_submission_csv(predictions, output_path)

    print("Submission generation completed successfully.")
