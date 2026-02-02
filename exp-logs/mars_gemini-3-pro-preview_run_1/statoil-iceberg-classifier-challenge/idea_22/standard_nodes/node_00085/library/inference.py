import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import setup_logger
from library.dataset import get_dataloaders
from library.model import AngleGatedResNet

logger = setup_logger()


def predict_with_tta(model, images, angles):
    """
    Applies Klein Four-Group Test-Time Augmentation (TTA) to the input batch.

    The four views are:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip
    4. Rotate 180 (equivalent to Horizontal + Vertical Flip)

    Args:
        model (nn.Module): The trained model.
        images (torch.Tensor): Batch of images (B, C, H, W).
        angles (torch.Tensor): Batch of incidence angles (B,).

    Returns:
        torch.Tensor: Averaged probabilities for the batch (B, 1).
    """
    # 1. Original
    logits_1 = model(images, angles)
    probs_1 = torch.sigmoid(logits_1)

    # 2. Horizontal Flip (Flip along width axis, dim 3)
    images_h = torch.flip(images, dims=[3])
    logits_2 = model(images_h, angles)
    probs_2 = torch.sigmoid(logits_2)

    # 3. Vertical Flip (Flip along height axis, dim 2)
    images_v = torch.flip(images, dims=[2])
    logits_3 = model(images_v, angles)
    probs_3 = torch.sigmoid(logits_3)

    # 4. Rotate 180 (Flip along both height and width)
    images_r180 = torch.flip(images, dims=[2, 3])
    logits_4 = model(images_r180, angles)
    probs_4 = torch.sigmoid(logits_4)

    # Average the probabilities
    avg_probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0

    return avg_probs


def generate_submission(num_models=5):
    """
    Generates the submission file by ensembling predictions from multiple SWA models.

    Steps:
    1. Loads the test dataloader.
    2. Iterates through each saved SWA checkpoint.
    3. Performs inference with TTA for each model.
    4. Aggregates predictions (averaging) across all models.
    5. Saves the result to submission.csv.

    Args:
        num_models (int): Number of independent SWA models to ensemble.
    """
    logger.info("Starting Inference and Submission Generation...")

    device = torch.device(Config.DEVICE)
    submission_dir = os.path.join(Config.WORKING_DIR, "submission")
    os.makedirs(submission_dir, exist_ok=True)
    checkpoints_dir = os.path.join(Config.WORKING_DIR, "checkpoints")

    # Load Test Data
    # load_cache=True ensures we use the pre-processed numpy arrays if available
    test_loader = get_dataloaders(phase="test", load_cache=True)

    # Variables to store ensemble results
    ensemble_probs = None
    ids_list = []

    models_found = 0

    for i in range(num_models):
        checkpoint_filename = f"swa_model_{i}.pth"
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_filename)

        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_filename} not found. Skipping.")
            continue

        logger.info(
            f"Processing with Model {i+1}/{num_models} ({checkpoint_filename})..."
        )

        # Initialize and load model
        model = AngleGatedResNet().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        model_probs_list = []
        current_ids_list = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                angles = batch["inc_angle"].to(device)
                batch_ids = batch["id"]

                # Predict with TTA
                batch_probs = predict_with_tta(model, images, angles)

                # Store results
                model_probs_list.append(batch_probs.cpu().numpy())

                # Collect IDs only during the first model's pass
                if i == 0:
                    current_ids_list.extend(batch_ids)

        # Concatenate all batches for this model
        full_model_probs = np.concatenate(
            model_probs_list, axis=0
        )  # Shape: (N_test, 1)

        # Accumulate into ensemble
        if ensemble_probs is None:
            ensemble_probs = full_model_probs
            ids_list = current_ids_list
        else:
            ensemble_probs += full_model_probs

        models_found += 1

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    if models_found == 0:
        logger.error(
            "No models were found for inference. Aborting submission generation."
        )
        return

    # Compute Arithmetic Mean of the Ensemble
    final_probs = ensemble_probs / models_found

    # Flatten to 1D array
    final_probs = final_probs.ravel()

    # Create DataFrame
    df_submission = pd.DataFrame({"id": ids_list, "is_iceberg": final_probs})

    # Ensure correct column order
    df_submission = df_submission[["id", "is_iceberg"]]

    # Save to CSV
    save_path = os.path.join(submission_dir, "submission.csv")
    df_submission.to_csv(save_path, index=False)

    logger.info(f"Submission generated successfully with {models_found} models.")
    logger.info(f"Saved to: {save_path}")
    logger.info(f"Number of predictions: {len(df_submission)}")
    logger.info("Head of submission:")
    logger.info(f"\n{df_submission.head()}")
