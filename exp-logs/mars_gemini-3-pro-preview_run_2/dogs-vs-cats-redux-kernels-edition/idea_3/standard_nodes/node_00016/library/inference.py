import os
import torch
import pandas as pd
import numpy as np
from library import config
from library import model as lib_model
from library import data
from library import utils


def predict_tta(model, loader, device):
    """
    Performs Test-Time Augmentation (TTA) by averaging predictions
    on original and horizontally flipped images.

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): The test data loader.
        device (str): Device to run inference on.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and 'label' (probability).
    """
    model.eval()
    results = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # 1. Forward pass original
            # Model outputs flattened logits or (B, 1) depending on architecture
            logits_orig = model(images).view(-1)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Forward pass flipped (TTA)
            # Flip width dimension (B, C, H, W) -> dim 3
            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped).view(-1)
            probs_flip = torch.sigmoid(logits_flip)

            # Average probabilities
            probs_avg = (probs_orig + probs_flip) / 2.0

            # Store results
            # Move to CPU and convert to numpy
            ids_np = ids.cpu().numpy()
            probs_np = probs_avg.cpu().numpy()

            for img_id, prob in zip(ids_np, probs_np):
                results.append({"id": img_id, "label": prob})

    df = pd.DataFrame(results)
    # Ensure ID is integer and sort
    df["id"] = df["id"].astype(int)
    df = df.sort_values("id").reset_index(drop=True)

    return df


def run_inference(
    checkpoint_name="model_best.pth",
    output_path=config.SUBMISSION_PATH,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    device=config.DEVICE,
    debug_subset_size=None,
):
    """
    Main function to load model, generate predictions with TTA, and save submission.

    Args:
        checkpoint_name (str): Name of the checkpoint file in config.CHECKPOINT_DIR.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        device (str): Device to use.
        debug_subset_size (int, optional): Limit dataset size for debugging.
    """
    # 1. Setup Data
    # We only need the test loader
    _, _, test_loader = data.get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        debug_subset_size=debug_subset_size,
    )

    # 2. Setup Model
    model = lib_model.create_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=False,  # No need to download pretrained weights, we load checkpoint
    )

    # 3. Load Checkpoint
    # utils.load_checkpoint loads from config.CHECKPOINT_DIR
    try:
        checkpoint = utils.load_checkpoint(checkpoint_name)
        # Checkpoint contains 'state_dict'
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {checkpoint_name}")
    except FileNotFoundError:
        print(f"Checkpoint {checkpoint_name} not found. Ensure training has run.")
        return

    model.to(device)

    # 4. Predict with TTA
    print("Starting TTA inference...")
    submission_df = predict_tta(model, test_loader, device)

    # 5. Save Submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(submission_df.head())
