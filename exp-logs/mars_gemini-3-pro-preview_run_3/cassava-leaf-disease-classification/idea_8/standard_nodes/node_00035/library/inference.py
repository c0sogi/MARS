import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import get_device, get_logger
from library.models import get_model
from library.data import get_dataloaders


def predict_with_tta(model, loader, device, tta_steps=1):
    """
    Generates predictions using Test-Time Augmentation (TTA).
    Averages Softmax probabilities across original and transformed views.

    Args:
        model: The PyTorch model to evaluate.
        loader: DataLoader for the test set.
        device: The computation device (CPU/GPU).
        tta_steps (int): Number of TTA views to use.
                         1=Original, 2=Original+HFlip,
                         3=Original+HFlip+VFlip, 4=Original+HFlip+VFlip+Transpose.

    Returns:
        all_probs (np.ndarray): Array of shape (N, Num_Classes) with averaged probabilities.
        all_ids (list): List of image IDs corresponding to the predictions.
    """
    model.eval()
    all_probs = []
    all_ids = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for images, ids in tqdm(loader, desc="Inference", leave=False):
            images = images.to(device)

            # Collect logits from requested views
            logits_list = []

            # View 1: Original
            logits_list.append(model(images))

            # View 2: Horizontal Flip
            if tta_steps >= 2:
                logits_list.append(model(torch.flip(images, dims=[3])))

            # View 3: Vertical Flip
            if tta_steps >= 3:
                logits_list.append(model(torch.flip(images, dims=[2])))

            # View 4: Transpose (Swap H and W)
            if tta_steps >= 4:
                logits_list.append(model(torch.transpose(images, 2, 3)))

            # Compute Softmax for each view and stack
            # Shape: (Num_Views, Batch_Size, Num_Classes)
            probs_stack = torch.stack([F.softmax(l, dim=1) for l in logits_list])

            # Average across views
            # Shape: (Batch_Size, Num_Classes)
            avg_probs = probs_stack.mean(dim=0)

            all_probs.append(avg_probs.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_probs), all_ids


def generate_ensemble_predictions(model_a_path, model_b_path):
    """
    Loads Model A and Model B, performs inference with TTA, ensembles the results,
    and saves the submission file.

    Args:
        model_a_path (str): Path to the saved checkpoint for Model A (ViT).
        model_b_path (str): Path to the saved checkpoint for Model B (EfficientNet).
    """
    # Initialize Configuration and Logging
    cfg = Config()
    device = get_device()
    log_path = os.path.join(cfg.WORKING_DIR, "inference.log")
    logger = get_logger(log_path)

    logger.info(f"Starting Ensemble Inference on device: {device}")

    # Load Test Data
    # get_dataloaders returns (train, val, test), we only need test
    _, _, test_loader = get_dataloaders(cfg)
    logger.info(f"Test Data Loaded. Samples: {len(test_loader.dataset)}")

    # ====================================================
    # Model A Inference (Global Expert: ViT)
    # ====================================================
    logger.info(f"--- Processing Model A: {cfg.MODEL_A_NAME} ---")
    model_a = get_model(cfg, cfg.MODEL_A_NAME)

    if os.path.exists(model_a_path):
        logger.info(f"Loading weights from {model_a_path}")
        state_dict = torch.load(model_a_path, map_location=device)
        model_a.load_state_dict(state_dict)
    else:
        logger.warning(
            f"Checkpoint not found at {model_a_path}. Using random weights (Warning!)."
        )

    model_a.to(device)

    # Predict with TTA
    probs_a, image_ids = predict_with_tta(
        model_a, test_loader, device, tta_steps=cfg.TTA_STEPS if cfg.USE_TTA else 1
    )

    # Clean up Model A to free GPU memory
    del model_a
    torch.cuda.empty_cache()

    # ====================================================
    # Model B Inference (Local Expert: EfficientNet)
    # ====================================================
    logger.info(f"--- Processing Model B: {cfg.MODEL_B_NAME} ---")
    model_b = get_model(cfg, cfg.MODEL_B_NAME)

    if os.path.exists(model_b_path):
        logger.info(f"Loading weights from {model_b_path}")
        state_dict = torch.load(model_b_path, map_location=device)
        model_b.load_state_dict(state_dict)
    else:
        logger.warning(
            f"Checkpoint not found at {model_b_path}. Using random weights (Warning!)."
        )

    model_b.to(device)

    # Predict with TTA
    probs_b, _ = predict_with_tta(
        model_b, test_loader, device, tta_steps=cfg.TTA_STEPS if cfg.USE_TTA else 1
    )

    # Clean up Model B
    del model_b
    torch.cuda.empty_cache()

    # ====================================================
    # Ensemble & Submission
    # ====================================================
    logger.info("--- Generating Ensemble Predictions ---")

    # Average probabilities (Soft Voting)
    final_probs = (probs_a + probs_b) / 2.0

    # Convert to class labels
    final_preds = np.argmax(final_probs, axis=1)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"image_id": image_ids, "label": final_preds})

    # Save to CSV
    save_path = cfg.SUBMISSION_PATH
    df_sub.to_csv(save_path, index=False)

    logger.info(f"Submission saved to {save_path}")
    logger.info(f"Submission shape: {df_sub.shape}")
    logger.info("Inference completed successfully.")
