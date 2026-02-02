import os
import torch
import numpy as np
import pandas as pd
from library.models import CactusRepVGG_MTL
from library.config import Config


def reparameterize_model(model):
    """
    Prepares a model for inference.
    If the model is a RepVGG variant, it fuses the structural branches
    (Conv3x3, 1x1, Identity) into a single 3x3 convolution for faster inference.

    Args:
        model (nn.Module): The trained model.

    Returns:
        nn.Module: The reparameterized model in eval mode.
    """
    if isinstance(model, CactusRepVGG_MTL):
        # Only switch if not already deployed
        if not model.deploy:
            model.switch_to_deploy()

    model.eval()
    return model


def predict_with_tta(model, loader, device):
    """
    Generates predictions using 4-view Test Time Augmentation (TTA).
    Views: Original, Horizontal Flip, Vertical Flip, 180-degree Rotation.

    Args:
        model (nn.Module): The model to use for prediction.
        loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.

    Returns:
        np.ndarray: Array of predicted probabilities (N,).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _, film_feats, _ in loader:
            images = images.to(device)
            film_feats = film_feats.to(device)

            # Create 4 views
            # View 0: Original
            v0 = images
            # View 1: Horizontal Flip
            v1 = torch.flip(images, dims=[3])
            # View 2: Vertical Flip
            v2 = torch.flip(images, dims=[2])
            # View 3: 180 Rotation (Horizontal + Vertical Flip)
            v3 = torch.flip(images, dims=[2, 3])

            # Forward pass for all views
            # Note: model returns (logits, aux_pred). We only need logits.
            # We apply sigmoid to convert logits to probabilities.
            p0 = torch.sigmoid(model(v0, film_feats)[0])
            p1 = torch.sigmoid(model(v1, film_feats)[0])
            p2 = torch.sigmoid(model(v2, film_feats)[0])
            p3 = torch.sigmoid(model(v3, film_feats)[0])

            # Average predictions across views
            avg_p = (p0 + p1 + p2 + p3) / 4.0

            preds.extend(avg_p.cpu().numpy().flatten())

    return np.array(preds)


def generate_submission(models, test_loader, output_path=None):
    """
    Generates a submission file by ensemble averaging predictions from multiple models.

    Args:
        models (list): List of loaded PyTorch models.
        test_loader (DataLoader): DataLoader for the test set.
        output_path (str, optional): Path to save the submission CSV.
                                     Defaults to Config.SUBMISSION_DIR/submission.csv.
    """
    if output_path is None:
        output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    device = Config.DEVICE
    print(f"Generating submission with {len(models)} models on {device}...")

    # 1. Load Test Metadata to get IDs
    # The loader is sequential (shuffle=False), so order matches metadata
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if Config.DEBUG:
        test_df = test_df.iloc[: Config.DEBUG_SUBSET_SIZE]

    test_ids = test_df["id"].values

    # 2. Accumulate predictions
    final_preds = np.zeros(len(test_ids))

    for i, model in enumerate(models):
        # Ensure model is optimized for inference
        model = reparameterize_model(model)
        model.to(device)

        # Run TTA Inference
        model_preds = predict_with_tta(model, test_loader, device)

        # Add to ensemble (Soft Voting)
        final_preds += model_preds

    # Average across models
    final_preds /= len(models)

    # 3. Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})

    # 4. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    return submission_df
