import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.model import RepVGGBlock


def convert_to_deploy(model):
    """
    Iterates through the trained model and switches RepVGGBlocks to deployment mode
    by fusing the multi-branch structure into a single convolution.

    Args:
        model (nn.Module): The trained PyTorch model.

    Returns:
        model (nn.Module): The re-parameterized model.
    """
    # Use the model's built-in reparameterize method if available
    if hasattr(model, "reparameterize"):
        model.reparameterize()
    else:
        # Fallback: Manual iteration through modules
        for module in model.modules():
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()

    return model


def predict_tta(model, test_loader, device=Config.DEVICE):
    """
    Performs inference using 4-view Test Time Augmentation (TTA).
    Views: Original, Horizontal Flip, Vertical Flip, Rotate 180.

    Args:
        model (nn.Module): The re-parameterized model.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Computation device.

    Returns:
        np.array: Flattened array of predicted probabilities.
    """
    model.eval()
    model.to(device)

    all_preds = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # View 1: Original
            logits1 = model(images)
            probs1 = torch.sigmoid(logits1)

            # View 2: Horizontal Flip (axis 3)
            logits2 = model(torch.flip(images, [3]))
            probs2 = torch.sigmoid(logits2)

            # View 3: Vertical Flip (axis 2)
            logits3 = model(torch.flip(images, [2]))
            probs3 = torch.sigmoid(logits3)

            # View 4: Rotate 180 (Horizontal + Vertical Flip)
            logits4 = model(torch.flip(images, [2, 3]))
            probs4 = torch.sigmoid(logits4)

            # Average predictions across all views
            avg_probs = (probs1 + probs2 + probs3 + probs4) / 4.0

            all_preds.append(avg_probs.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def generate_submission(
    model, test_loader, device=Config.DEVICE, output_path=Config.SUBMISSION_PATH
):
    """
    Orchestrates the conversion to deploy mode, TTA prediction, and CSV generation.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Computation device.
        output_path (str): Path to save the submission CSV.
    """
    # 1. Optimize model structure for inference (Reparameterization)
    print("Converting model to deploy mode (fusing blocks)...")
    convert_to_deploy(model)

    # 2. Generate predictions with TTA
    print("Generating predictions with TTA...")
    predictions = predict_tta(model, test_loader, device)

    # 3. Load metadata to map IDs
    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_METADATA}")

    df_test = pd.read_csv(Config.TEST_METADATA)

    # Validation check
    if len(predictions) != len(df_test):
        print(
            f"Warning: Prediction count ({len(predictions)}) does not match Test set size ({len(df_test)})"
        )

    # 4. Assign predictions and save
    df_test["has_cactus"] = predictions

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save only required columns: id, has_cactus
    df_test[["id", "has_cactus"]].to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
