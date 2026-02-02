import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from library.config import CFG, seed_everything
from library.utils import get_logger
from library.models import CassavaClassifier


def predict_fn(model, loader, device, tta_steps=1):
    """
    Runs inference on a loader using the given model with Test-Time Augmentation (TTA).

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): The data loader for inference.
        device (torch.device): The device to run inference on.
        tta_steps (int): Number of TTA views.
                         1 = Original only
                         2 = Original + Horizontal Flip
                         3 = Original + Horizontal Flip + Vertical Flip

    Returns:
        np.ndarray: Softmax probabilities of shape (N, num_classes)
    """
    model.eval()
    probs = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            batch_probs = []

            # 1. Original View
            out = model(images)
            batch_probs.append(torch.softmax(out, dim=1))

            # 2. Horizontal Flip
            if tta_steps >= 2:
                # N, C, H, W -> Flip W (dim 3)
                out_h = model(torch.flip(images, dims=[3]))
                batch_probs.append(torch.softmax(out_h, dim=1))

            # 3. Vertical Flip
            if tta_steps >= 3:
                # N, C, H, W -> Flip H (dim 2)
                out_v = model(torch.flip(images, dims=[2]))
                batch_probs.append(torch.softmax(out_v, dim=1))

            # Average predictions across TTA views
            avg_probs = torch.stack(batch_probs).mean(dim=0)
            probs.append(avg_probs.cpu().numpy())

    return np.concatenate(probs)


def generate_submission(model_path, test_loader):
    """
    Loads a trained model, runs inference with TTA, and saves the submission file.

    Args:
        model_path (str): Path to the best checkpoint.
        test_loader (DataLoader): DataLoader for the test set.
    """
    # Ensure reproducibility
    seed_everything(CFG.seed)
    device = CFG.device

    # Setup Logging
    os.makedirs(CFG.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(CFG.output_dir, "inference.log"))

    logger.info("Starting Inference Pipeline...")
    logger.info(f"TTA Steps: {CFG.tta_steps}")

    # -------------------------------------------------------------------------
    # Model Inference
    # -------------------------------------------------------------------------
    logger.info(f"--- Processing Model: {CFG.model_name} ---")
    if not os.path.exists(model_path):
        logger.error(f"Checkpoint not found at {model_path}")
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    # Initialize Model
    model = CassavaClassifier(CFG.model_name, CFG.num_classes, pretrained=False)

    # Load Weights
    logger.info(f"Loading weights from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    # Predict
    logger.info("Running inference...")
    final_probs = predict_fn(model, test_loader, device, tta_steps=CFG.tta_steps)

    # Cleanup
    del model, checkpoint, state_dict
    torch.cuda.empty_cache()
    logger.info("Inference complete. Memory cleared.")

    final_preds = np.argmax(final_probs, axis=1)

    logger.info(f"Generated {len(final_preds)} predictions.")

    # Load Test Metadata to ensure correct ID mapping
    if os.path.exists(CFG.test_csv):
        test_df = pd.read_csv(CFG.test_csv)
    else:
        # Fallback to sample submission if metadata is missing (unlikely per spec)
        logger.warning("Test metadata not found. Falling back to sample_submission.csv")
        test_df = pd.read_csv(os.path.join(CFG.input_root, "sample_submission.csv"))

    # Assign labels
    test_df["label"] = final_preds

    # Prepare submission dataframe
    submission_df = test_df[["image_id", "label"]]

    # Save to disk
    os.makedirs(os.path.dirname(CFG.submission_file), exist_ok=True)
    submission_df.to_csv(CFG.submission_file, index=False)

    logger.info(f"Submission file saved successfully to: {CFG.submission_file}")
    logger.info("Ensemble Inference Pipeline Finished.")
