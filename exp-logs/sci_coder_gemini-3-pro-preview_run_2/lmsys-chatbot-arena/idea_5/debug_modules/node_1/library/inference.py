import os
import gc
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

from library.config import Config
from library.utils import get_logger
from library.data import load_data, get_test_dataloader
from library.model import SiameseModel

# Initialize logger
logger = get_logger("inference")


def generate_submission(load_cached_data=True):
    """
    Generates the submission file by running inference using an ensemble of trained models.

    Args:
        load_cached_data (bool): Whether to load preprocessed data from cache.
    """
    logger.info("Starting submission generation...")

    device = torch.device(Config.DEVICE)

    # 1. Load Data
    # load_data returns (train_df, test_df). We only need test_df.
    _, test_df = load_data(load_cached_data=load_cached_data)

    # 2. Prepare DataLoader
    logger.info("Initializing tokenizer and dataloader...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    test_loader = get_test_dataloader(
        test_df, tokenizer, batch_size=Config.VALID_BATCH_SIZE
    )

    # 3. Ensemble Inference
    # Array to store accumulated probabilities: [N_samples, 3]
    ensemble_probs = np.zeros((len(test_df), 3), dtype=np.float32)
    models_found = 0

    for fold_idx in range(Config.N_FOLDS):
        model_path = os.path.join(
            Config.MODEL_OUTPUT_DIR, f"best_model_fold_{fold_idx}.pth"
        )

        if not os.path.exists(model_path):
            logger.warning(
                f"Model checkpoint not found at {model_path}. Skipping fold {fold_idx}."
            )
            continue

        logger.info(f"Processing Fold {fold_idx} using checkpoint: {model_path}")

        # Initialize model and load weights
        model = SiameseModel()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        fold_probs = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids_a = batch["input_ids_a"].to(device)
                attention_mask_a = batch["attention_mask_a"].to(device)
                input_ids_b = batch["input_ids_b"].to(device)
                attention_mask_b = batch["attention_mask_b"].to(device)
                meta_features = batch["meta_features"].to(device)

                # Forward pass
                logits = model(
                    input_ids_a,
                    attention_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    meta_features,
                )

                # Apply Softmax to get probabilities
                probs = F.softmax(logits, dim=1)
                fold_probs.append(probs.cpu().numpy())

        # Concatenate batches for this fold
        fold_probs = np.concatenate(fold_probs, axis=0)

        # Add to ensemble accumulator
        ensemble_probs += fold_probs
        models_found += 1

        # Cleanup to free GPU memory
        del model, state_dict, fold_probs
        torch.cuda.empty_cache()
        gc.collect()

    if models_found == 0:
        logger.error("No models were loaded. Cannot generate submission.")
        return

    # 4. Average Probabilities
    avg_probs = ensemble_probs / models_found

    # 5. Create Submission DataFrame
    logger.info("Creating submission DataFrame...")
    submission_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "winner_model_a": avg_probs[:, 0],
            "winner_model_b": avg_probs[:, 1],
            "winner_tie": avg_probs[:, 2],
        }
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Print first few rows for verification
    logger.info("Head of submission file:")
    logger.info(submission_df.head().to_string())
