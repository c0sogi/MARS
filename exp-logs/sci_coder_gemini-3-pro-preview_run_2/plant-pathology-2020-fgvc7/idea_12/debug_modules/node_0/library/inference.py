import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import get_data, AppleDataset
from library.model import AppleDiseaseModel
from library.utils import get_logger, seed_everything


def predict_tta(model, loader, device):
    """
    Generates predictions using Test Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for test data.
        device (torch.device): Computation device.

    Returns:
        np.ndarray: Array of predicted probabilities. Shape: (N_samples, N_classes).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Forward pass: Original
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Forward pass: Horizontal Flip (TTA)
            if Config.TTA_FLIP:
                images_flipped = torch.flip(
                    images, dims=[3]
                )  # Flip width dimension (N, C, H, W)
                logits_flip = model(images_flipped)
                probs_flip = torch.sigmoid(logits_flip)

                # Average probabilities
                probs = (probs_orig + probs_flip) / 2.0
            else:
                probs = probs_orig

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def rank_average(predictions_list):
    """
    Combines predictions from multiple models using Rank Averaging.

    Strategy:
    1. Convert probabilities to ranks for each model to remove calibration bias.
    2. Normalize ranks to [0, 1].
    3. Average the normalized ranks.

    Args:
        predictions_list (list of np.ndarray): List of prediction arrays from different models.
                                               Each array has shape (N_samples, N_classes).

    Returns:
        np.ndarray: Aggregated rank scores. Shape: (N_samples, N_classes).
    """
    if not predictions_list:
        return None

    n_samples, n_classes = predictions_list[0].shape
    aggregated_ranks = np.zeros((n_samples, n_classes), dtype=np.float32)

    for preds in predictions_list:
        # Compute ranks for each column (class)
        # argsort().argsort() gives the rank (0 to N-1)
        ranks = np.zeros_like(preds)
        for c in range(n_classes):
            # We want rank 0 to be lowest prob, rank N-1 to be highest
            # method='ordinal' equivalent via double argsort
            ranks[:, c] = np.argsort(np.argsort(preds[:, c]))

        # Normalize to [0, 1]
        normalized_ranks = ranks / (n_samples - 1 + 1e-6)
        aggregated_ranks += normalized_ranks

    # Average across models
    avg_ranks = aggregated_ranks / len(predictions_list)
    return avg_ranks


def reconstruct_probabilities(scores):
    """
    Maps aggregated binary scores (Rust, Scab) to the 4-class format.

    Logic:
    - Healthy = (1 - Rust) * (1 - Scab)
    - Rust = Rust * (1 - Scab)
    - Scab = (1 - Rust) * Scab
    - Multiple = Rust * Scab

    Args:
        scores (np.ndarray): Aggregated scores. Shape: (N_samples, 2).
                             Column 0: Rust Score, Column 1: Scab Score.

    Returns:
        np.ndarray: Final probabilities. Shape: (N_samples, 4).
        Columns: [healthy, multiple_diseases, rust, scab]
    """
    S_r = scores[:, 0]  # Rust Score
    S_s = scores[:, 1]  # Scab Score

    # Calculate class probabilities
    p_healthy = (1 - S_r) * (1 - S_s)
    p_rust = S_r * (1 - S_s)
    p_scab = (1 - S_r) * S_s
    p_multiple = S_r * S_s

    # Stack columns
    # Order must match sample_submission: healthy, multiple_diseases, rust, scab
    final_probs = np.stack([p_healthy, p_multiple, p_rust, p_scab], axis=1)

    # Normalize rows to sum to 1 (handling potential floating point drift)
    row_sums = final_probs.sum(axis=1, keepdims=True)
    final_probs = final_probs / (row_sums + 1e-6)

    return final_probs


def run_inference():
    """
    Main inference orchestration function.
    """
    seed_everything(Config.SEED)
    logger = get_logger("inference")

    logger.info("Starting Inference Pipeline...")

    # 1. Load Test Data
    test_df = get_data("test")
    logger.info(f"Test Data Loaded: {len(test_df)} samples")

    all_model_preds = []

    # 2. Iterate through all Models and Folds
    for model_cfg in Config.MODELS:
        model_name = model_cfg["name"]
        safe_model_name = model_name.replace(".", "_")
        img_size = model_cfg["img_size"]

        # Setup Dataset and Loader
        test_dataset = AppleDataset(test_df, img_size=img_size, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=model_cfg["batch_size"],
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        for fold in range(Config.FOLDS):
            # Construct model path
            # We prioritize the 'best_model' which might have been updated by SWA logic in training
            checkpoint_path = os.path.join(
                Config.WORKING_DIR, f"best_model_{safe_model_name}_fold_{fold}.pth"
            )

            if not os.path.exists(checkpoint_path):
                logger.warning(f"Checkpoint not found: {checkpoint_path}. Skipping...")
                continue

            logger.info(f"Predicting with {model_name} (Fold {fold})...")

            # Initialize Model
            model = AppleDiseaseModel(
                model_name=model_name,
                pretrained=False,  # Weights loaded manually
                num_classes=Config.NUM_TARGETS,
                gem_p=model_cfg["gem_p"],
                num_msd=model_cfg["num_msd"],
                msd_dropout=model_cfg["msd_dropout"],
            ).to(Config.DEVICE)

            # Load Weights
            state_dict = torch.load(checkpoint_path, map_location=Config.DEVICE)
            model.load_state_dict(state_dict)

            # Generate Predictions (with TTA)
            preds = predict_tta(model, test_loader, Config.DEVICE)
            all_model_preds.append(preds)

    if not all_model_preds:
        raise RuntimeError(
            "No predictions generated. Check if model checkpoints exist."
        )

    # 3. Ensemble Aggregation (Rank Averaging)
    logger.info(
        f"Aggregating predictions from {len(all_model_preds)} models using Rank Averaging..."
    )
    aggregated_scores = rank_average(all_model_preds)

    # 4. Reconstruct Final Probabilities
    logger.info("Reconstructing 4-class probabilities...")
    final_probs = reconstruct_probabilities(aggregated_scores)

    # 5. Create Submission File
    submission_df = pd.DataFrame(
        {
            "image_id": test_df["image_id"],
            "healthy": final_probs[:, 0],
            "multiple_diseases": final_probs[:, 1],
            "rust": final_probs[:, 2],
            "scab": final_probs[:, 3],
        }
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Print first few rows for verification
    print(submission_df.head())
