import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    IDEA_DIR,
    SUBMISSION_PATH,
    N_FOLDS,
    SEED,
)
from library.utils import seed_everything, get_logger
from library.dataset import BraTSDataset, get_transforms
from library.model import CASIVNet


def predict_test_set(
    model_dir=IDEA_DIR,
    output_path=SUBMISSION_PATH,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
):
    """
    Loads the ensemble of trained models, generates predictions for the test set,
    and saves the submission file.

    Args:
        model_dir (str): Directory containing the trained model checkpoints.
        output_path (str): Path to save the submission CSV.
        device (torch.device): Device to run inference on.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker processes for data loading.
    """
    # Ensure reproducibility
    seed_everything(SEED)

    # Setup logging
    logger = get_logger(os.path.join(model_dir, "inference.log"))
    logger.info(f"Starting inference on device: {device}")

    # 1. Load Test Dataset
    # We use the 'test' split which utilizes test_metadata.csv
    # We use 'test' transforms (no augmentation, just tensor conversion)
    test_ds = BraTSDataset(split="test", transform=get_transforms("test"))

    if len(test_ds) == 0:
        logger.warning("Test dataset is empty. Creating empty submission.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(columns=["BraTS21ID", "MGMT_value"]).to_csv(
            output_path, index=False
        )
        return

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Test set size: {len(test_ds)} subjects")

    # 2. Load Ensemble Models
    models = []
    for fold in range(N_FOLDS):
        checkpoint_path = os.path.join(model_dir, f"best_model_fold{fold}.pth")

        if not os.path.exists(checkpoint_path):
            logger.warning(
                f"Checkpoint for fold {fold} not found at {checkpoint_path}. Skipping."
            )
            continue

        try:
            # Initialize architecture
            model = CASIVNet()
            model.to(device)

            # Load weights
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Handle case where checkpoint saves 'state_dict' key or just the dict
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
            logger.info(f"Loaded model for fold {fold}")

        except Exception as e:
            logger.error(f"Failed to load model for fold {fold}: {e}")

    if not models:
        logger.error("No models loaded. Cannot perform inference.")
        return

    # 3. Run Inference
    # We accumulate probabilities for each subject ID
    # results[braTS21ID] = sum_of_probabilities
    results = {}

    logger.info("Running prediction loop...")

    # Disable gradient calculation for inference
    with torch.no_grad():
        for i, model in enumerate(models):
            logger.info(f"Predicting with model {i+1}/{len(models)}...")

            for images, ids in test_loader:
                images = images.to(device)

                # Forward pass
                logits = model(images)
                probs = torch.sigmoid(logits)

                # Move to CPU
                probs = probs.cpu().numpy().flatten()
                ids = ids.numpy()

                # Accumulate
                for pid, prob in zip(ids, probs):
                    if pid not in results:
                        results[pid] = 0.0
                    results[pid] += prob

    # 4. Average and Format
    final_preds = []
    num_models = len(models)

    for pid in sorted(results.keys()):
        avg_prob = results[pid] / num_models
        final_preds.append({"BraTS21ID": pid, "MGMT_value": avg_prob})

    df_submission = pd.DataFrame(final_preds)

    # 5. Save Submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_submission.to_csv(output_path, index=False)

    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Head of submission:\n{df_submission.head()}")
