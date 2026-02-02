import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import get_model


def predict_tta(model, inputs, device):
    """
    Performs inference using Test-Time Augmentation (TTA).
    Averages the probabilities of the original image and a horizontally flipped version.

    Args:
        model (torch.nn.Module): The trained model.
        inputs (torch.Tensor): Batch of images (B, C, H, W).
        device (torch.device): Compute device.

    Returns:
        np.ndarray: Averaged probability predictions (B, Num_Classes).
    """
    # Ensure inputs are on the correct device
    inputs = inputs.to(device)

    # 1. Forward pass on original images
    with torch.no_grad():
        outputs_orig = model(inputs)
        probs_orig = torch.softmax(outputs_orig, dim=1)

    # 2. Forward pass on horizontally flipped images
    # Flip along the width dimension (dim=3 for NCHW format)
    inputs_flipped = torch.flip(inputs, dims=[3])
    with torch.no_grad():
        outputs_flipped = model(inputs_flipped)
        probs_flipped = torch.softmax(outputs_flipped, dim=1)

    # 3. Average the probabilities
    avg_probs = (probs_orig + probs_flipped) / 2.0

    return avg_probs.cpu().numpy()


def generate_submission(model_path=None, batch_size=Config.BATCH_SIZE):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model_path (str, optional): Path to the trained model weights.
                                    Defaults to best_model.pth in working dir.
        batch_size (int): Batch size for inference.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    if model_path is None:
        model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting submission generation using model: {model_path}")

    # 1. Load Data
    # We load cached data to ensure consistency with training metadata
    dataloaders, class_names = get_dataloaders(
        batch_size=batch_size, num_workers=Config.NUM_WORKERS, load_cached_data=True
    )
    test_loader = dataloaders["test"]

    # 2. Load Model
    print(f"Initializing model: {Config.MODEL_NAME}")
    model = get_model(num_classes=Config.NUM_CLASSES, pretrained=False)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    # Handle case where checkpoint saves 'model_state_dict' or just the state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # 3. Inference Loop
    all_ids = []
    all_probs = []

    print("Running inference with TTA...")
    # Using tqdm for progress tracking is helpful, but we will keep it silent if required
    # or minimal. The prompt asks not to print progress bars, so we iterate directly.

    for inputs, ids in test_loader:
        # Predict with TTA
        batch_probs = predict_tta(model, inputs, device)

        all_probs.append(batch_probs)
        all_ids.extend(ids)

    # Concatenate all predictions
    final_probs = np.concatenate(all_probs, axis=0)

    # 4. Create Submission DataFrame
    # The format requires: id, breed1, breed2, ...
    # class_names are already sorted alphabetically from get_dataloaders

    df_submission = pd.DataFrame(final_probs, columns=class_names)
    df_submission.insert(0, "id", all_ids)

    # 5. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    save_path = Config.SUBMISSION_PATH
    df_submission.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print(f"Submission shape: {df_submission.shape}")
