import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import PathologyDataset, get_transforms
from library.model import get_model


def predict_with_tta(model, loader, device=Config.DEVICE):
    """
    Performs inference using 8-view Test Time Augmentation (TTA).
    The 8 views correspond to the Dihedral group D4:
    - Identity
    - Rotate 90, 180, 270
    - Horizontal Flip
    - Horizontal Flip + Rotate 90, 180, 270

    Args:
        model (nn.Module): The trained PyTorch model.
        loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.

    Returns:
        np.ndarray: Flattened array of predicted probabilities averaged across views.
    """
    model.eval()
    model.to(device)

    all_preds = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # Accumulator for probabilities
            # We sum probabilities from all views then divide by 8
            avg_probs = torch.zeros((batch_size, 1), device=device)

            # Define views
            views = []

            # 1. Original
            views.append(inputs)

            # 2-4. Rotations of Original (90, 180, 270)
            # dims=[2, 3] corresponds to H, W
            views.append(torch.rot90(inputs, 1, [2, 3]))
            views.append(torch.rot90(inputs, 2, [2, 3]))
            views.append(torch.rot90(inputs, 3, [2, 3]))

            # 5. Horizontal Flip
            inputs_flip = torch.flip(inputs, [3])
            views.append(inputs_flip)

            # 6-8. Rotations of Flipped (90, 180, 270)
            views.append(torch.rot90(inputs_flip, 1, [2, 3]))
            views.append(torch.rot90(inputs_flip, 2, [2, 3]))
            views.append(torch.rot90(inputs_flip, 3, [2, 3]))

            # Process all views
            for view in views:
                logits = model(view)
                probs = torch.sigmoid(logits)
                avg_probs += probs

            # Average over the 8 views
            avg_probs /= 8.0

            all_preds.append(avg_probs.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def run_inference(
    checkpoint_path=Config.MODEL_CHECKPOINT,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    device=Config.DEVICE,
):
    """
    Main entry point for inference. Loads model, runs TTA prediction, and saves submission.

    Args:
        checkpoint_path (str): Path to the saved model weights.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        device (str): Device to run on.
    """
    print(f"Initializing inference on {device}...")

    # 1. Load Data
    # 'test' mode uses metadata/test.csv and applies validation/test transforms (no augmentation)
    test_dataset = PathologyDataset(mode="test", transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Must be False to align with IDs
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    print(f"Test set size: {len(test_dataset)}")

    # 2. Load Model
    print("Loading model architecture...")
    model = get_model(
        model_name=Config.MODEL_NAME,
        pretrained=False,  # We load our own weights, no need to download ImageNet weights
        num_classes=Config.NUM_CLASSES,
    )

    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Checkpoint {checkpoint_path} not found. Predictions will be random/untrained."
        )

    # 3. Predict
    print("Starting 8-view TTA prediction...")
    predictions = predict_with_tta(model, test_loader, device)

    # 4. Save Submission
    # Retrieve IDs from the dataset dataframe.
    # Since shuffle=False, the order matches the predictions.
    ids = test_dataset.df["id"].values

    if len(ids) != len(predictions):
        raise ValueError(f"Mismatch: {len(ids)} IDs vs {len(predictions)} predictions.")

    df_submission = pd.DataFrame({"id": ids, "label": predictions})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
