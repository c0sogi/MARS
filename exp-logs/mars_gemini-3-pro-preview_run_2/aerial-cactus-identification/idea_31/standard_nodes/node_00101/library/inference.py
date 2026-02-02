import torch
import torchvision.transforms.functional as TF
import pandas as pd
import numpy as np
from library.config import Config


def predict(model, loader, device):
    """
    Performs inference on the test set using Test Time Augmentation (TTA).
    Generates predictions for original, horizontally flipped, and vertically flipped images,
    then averages the probabilities.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        loader (torch.utils.data.DataLoader): The test data loader.
        device (torch.device): The device to run inference on.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and 'has_cactus' columns with predicted probabilities.
    """
    model.eval()

    all_probs = []
    all_ids = []

    # Determine if TTA is enabled from Config
    use_tta = Config.TTA_ENABLED

    with torch.no_grad():
        for images, _, ids in loader:
            images = images.to(device)

            # 1. Prediction on Original Image
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            if use_tta:
                # 2. Prediction on Horizontally Flipped Image
                # TF.hflip works on tensors of shape (B, C, H, W)
                images_h = TF.hflip(images)
                logits_h = model(images_h)
                probs_h = torch.sigmoid(logits_h)

                # 3. Prediction on Vertically Flipped Image
                images_v = TF.vflip(images)
                logits_v = model(images_v)
                probs_v = torch.sigmoid(logits_v)

                # Average the probabilities
                # (Original + HFlip + VFlip) / 3
                batch_probs = (probs_orig + probs_h + probs_v) / 3.0
            else:
                batch_probs = probs_orig

            # Flatten to 1D array and store
            all_probs.append(batch_probs.cpu().numpy().flatten())
            all_ids.extend(ids)

    # Concatenate all batch predictions
    final_probs = np.concatenate(all_probs)

    # Create DataFrame
    submission_df = pd.DataFrame({"id": all_ids, "has_cactus": final_probs})

    return submission_df
