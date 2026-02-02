import numpy as np
import torch
from torch.cuda.amp import autocast
from library.config import Config


def predict_tta(model, loader, device):
    """
    Performs Test Time Augmentation (Original + Horizontal Flip) on the given dataset.

    Args:
        model (torch.nn.Module): The trained model to use for inference.
        loader (torch.utils.data.DataLoader): DataLoader providing the images.
        device (torch.device or str): The device to run the model on.

    Returns:
        np.ndarray: The averaged predicted probabilities for the dataset.
                    Shape: (num_samples, num_classes).
    """
    model.eval()
    all_preds = []

    # Ensure device is a torch.device object
    if isinstance(device, str):
        device = torch.device(device)

    with torch.no_grad():
        for batch in loader:
            # Unpack the batch; DogDataset returns (img, target, img_id)
            images = batch[0]
            images = images.to(device)

            # 1. Forward pass on original images
            with autocast():
                outputs_orig = model(images)
                probs_orig = torch.softmax(outputs_orig, dim=1)

            # 2. Forward pass on horizontally flipped images
            # Tensor shape is (Batch, Channels, Height, Width). Flip along Width (dim 3).
            images_flipped = torch.flip(images, dims=[3])

            with autocast():
                outputs_flipped = model(images_flipped)
                probs_flipped = torch.softmax(outputs_flipped, dim=1)

            # 3. Average the probabilities
            avg_probs = (probs_orig + probs_flipped) / 2.0

            all_preds.append(avg_probs.cpu().numpy())

    # Concatenate predictions from all batches
    if len(all_preds) > 0:
        return np.concatenate(all_preds, axis=0)
    else:
        return np.empty((0, Config.NUM_CLASSES))
