import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import CFG
from library.utils import get_logger, seed_everything
from library.data import CassavaDataset, get_transforms
from library.network import get_model


def predict_fold(fold, test_loader, device):
    """
    Generates predictions for a specific fold using the best saved model.
    Applies Test Time Augmentation (Horizontal Flip).

    Args:
        fold (int): The fold number (0-4).
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.

    Returns:
        torch.Tensor: Tensor of soft probabilities (N, num_classes).
    """
    # Initialize model architecture
    # pretrained=False because we are loading our own fine-tuned weights
    model = get_model(pretrained=False)

    # Construct weight path
    # Weights are expected in the output directory defined in CFG
    weight_path = os.path.join(CFG.output_dir, f"best_model_fold_{fold}.pth")

    if not os.path.exists(weight_path):
        print(
            f"Warning: Weight file not found for fold {fold} at {weight_path}. Skipping."
        )
        return None

    # Load weights
    try:
        checkpoint = torch.load(weight_path, map_location=device)

        # Handle various checkpoint formats (full state vs state_dict)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Clean state_dict keys if necessary (e.g., remove 'module.' prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Error loading weights for fold {fold}: {e}")
        return None

    # Set model to evaluation mode (disables dropout, stochastic depth, etc.)
    model.eval()

    probs_list = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # --- TTA Strategy ---
            # 1. Forward pass with original image
            output_orig = model(images)
            probs_orig = F.softmax(output_orig, dim=1)

            if CFG.tta:
                # 2. Forward pass with horizontally flipped image
                # Dim 3 is width (B, C, H, W)
                images_flip = torch.flip(images, dims=[3])
                output_flip = model(images_flip)
                probs_flip = F.softmax(output_flip, dim=1)

                # Average probabilities
                probs_avg = (probs_orig + probs_flip) / 2.0
            else:
                probs_avg = probs_orig

            probs_list.append(probs_avg.cpu())

    # Concatenate all batches to form the full prediction tensor for this fold
    return torch.cat(probs_list, dim=0)


def inference():
    """
    Main inference routine.
    Loads test metadata, performs ensemble inference across all folds,
    and saves the final submission CSV.
    """
    # Ensure reproducibility
    seed_everything(CFG.seed)

    # Setup logger
    logger = get_logger(os.path.join(CFG.working_dir, "inference.log"))
    logger.info("Starting Inference...")

    # 1. Load Test Metadata
    if not os.path.exists(CFG.test_csv):
        logger.error(f"Test metadata file not found at {CFG.test_csv}")
        return

    test_df = pd.read_csv(CFG.test_csv)
    logger.info(f"Loaded test metadata containing {len(test_df)} samples.")

    # 2. Prepare Data Loader
    # Use the Phase 2 image size (384) as the model was fine-tuned on this resolution
    transform = get_transforms("test", CFG.p2_img_size)
    test_ds = CassavaDataset(test_df, transform=transform)

    # We can use a slightly larger batch size for inference as we don't store gradients
    inference_batch_size = CFG.p2_batch_size * 2

    test_loader = DataLoader(
        test_ds,
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Run Inference for each fold
    fold_predictions = []

    for fold in range(CFG.n_folds):
        logger.info(f"Running inference for Fold {fold}...")
        fold_probs = predict_fold(fold, test_loader, CFG.device)

        if fold_probs is not None:
            fold_predictions.append(fold_probs)
        else:
            logger.warning(
                f"Skipping Fold {fold} due to missing weights or load error."
            )

    if not fold_predictions:
        logger.error("No valid predictions generated from any fold. Aborting.")
        return

    # 4. Ensemble Aggregation
    # Stack predictions: (num_folds, num_samples, num_classes)
    stacked_probs = torch.stack(fold_predictions)

    # Average across folds
    mean_probs = torch.mean(stacked_probs, dim=0)

    # Determine final labels (argmax)
    final_labels = torch.argmax(mean_probs, dim=1).numpy()

    # 5. Generate Submission File
    submission = pd.DataFrame({"image_id": test_df["image_id"], "label": final_labels})

    # Save to the experiment working directory
    submission.to_csv(CFG.submission_csv, index=False)
    logger.info(f"Submission saved to {CFG.submission_csv}")

    # Save to the specific submission directory required by the task
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    final_submission_path = os.path.join(submission_dir, "submission.csv")
    submission.to_csv(final_submission_path, index=False)
    logger.info(f"Submission also saved to {final_submission_path}")

    # Log sample
    logger.info("Submission Head:")
    logger.info(submission.head())
