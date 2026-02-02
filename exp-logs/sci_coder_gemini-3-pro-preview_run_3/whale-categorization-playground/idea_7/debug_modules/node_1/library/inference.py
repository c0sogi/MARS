import torch
import torch.nn.functional as F
import numpy as np
from library.config import Config


def extract_embeddings(model, dataloader, device):
    """
    Generates embeddings for a given dataset using a trained model.

    Features:
    - Applies Test-Time Augmentation (TTA) by averaging predictions of
      original and horizontally flipped images if enabled in Config.
    - Applies L2 normalization to the final embeddings to project them
      onto the hypersphere.
    - Handles both tensor targets (labels) and string targets (filenames).

    Args:
        model (torch.nn.Module): The trained neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the images.
        device (str): The device to run inference on (e.g., 'cuda', 'cpu').

    Returns:
        tuple: (embeddings, targets)
            - embeddings (np.ndarray): Feature matrix of shape (N_samples, Embedding_Dim).
            - targets (np.ndarray): Array of shape (N_samples,) containing labels or filenames.
    """
    model.eval()

    all_embeddings = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)

            # 1. Forward Pass (Original Images)
            features = model(images)

            # 2. Test-Time Augmentation (Horizontal Flip)
            if Config.TTA_ENABLED:
                # Flip images along the width axis (dim 3 for NCHW format)
                images_flipped = torch.flip(images, dims=[3])
                features_flipped = model(images_flipped)

                # Average the embeddings
                features = (features + features_flipped) / 2.0

            # 3. L2 Normalization
            # Essential for ArcFace/Cosine Similarity based retrieval
            features = F.normalize(features, p=2, dim=1)

            # 4. Collect Results
            all_embeddings.append(features.cpu().numpy())

            # Handle targets: they can be Tensors (labels) or tuples (filenames)
            if isinstance(targets, torch.Tensor):
                all_targets.append(targets.cpu().numpy())
            else:
                # Convert tuple/list of strings to numpy array
                all_targets.append(np.array(targets))

    # 5. Concatenate all batches into single arrays
    if len(all_embeddings) > 0:
        embeddings = np.vstack(all_embeddings)
        targets = np.concatenate(all_targets)
    else:
        embeddings = np.array([])
        targets = np.array([])

    return embeddings, targets
