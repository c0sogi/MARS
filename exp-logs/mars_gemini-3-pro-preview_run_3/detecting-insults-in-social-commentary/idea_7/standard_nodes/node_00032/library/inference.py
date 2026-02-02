import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

from library.config import Config
from library.utils import get_logger
from library.dataset import load_processed_data, get_dataloader
from library.model import InsultModel

# Initialize logger
logger = get_logger("inference")


def inference_fn(model, dataloader, device):
    """
    Performs inference on the provided dataloader using the given model.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        np.array: Array of predicted probabilities.
    """
    model.eval()
    preds = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Use Mixed Precision for efficiency
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(input_ids, attention_mask)
                # Apply sigmoid to get probabilities [0, 1]
                probs = torch.sigmoid(logits).squeeze(1)

            preds.append(probs.cpu().float().numpy())

    return np.concatenate(preds)


def run_inference():
    """
    Main function to run the inference pipeline.
    Loads test data, loads ensemble models, generates predictions, and saves submission.
    """
    logger.info("Starting Inference Pipeline...")

    # ==========================================
    # 1. Load Test Data
    # ==========================================
    # We force debug=False to ensure we generate predictions for the full test set
    df_test = load_processed_data(
        Config.TEST_PATH,
        "test_data.parquet",
        load_cached_data=True,
        debug=False,
    )
    logger.info(f"Loaded test data with {len(df_test)} samples.")

    # ==========================================
    # 2. Prepare DataLoader
    # ==========================================
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Use VALID_BATCH_SIZE for inference as it's usually larger than TRAIN_BATCH_SIZE
    test_loader = get_dataloader(
        df_test,
        tokenizer,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        max_len=Config.MAX_LEN,
        is_test=True,
    )

    device = torch.device(Config.DEVICE)

    # ==========================================
    # 3. Ensemble Inference
    # ==========================================
    # Array to store sum of predictions from all models
    final_preds = np.zeros(len(df_test))
    models_found = 0

    for seed in Config.SEEDS:
        weight_path = os.path.join(Config.OUTPUT_DIR, f"model_seed_{seed}.bin")

        if not os.path.exists(weight_path):
            logger.warning(
                f"Model weights not found for seed {seed} at {weight_path}. Skipping."
            )
            continue

        logger.info(f"Running inference with model seed: {seed}")

        # Initialize model
        model = InsultModel(Config.MODEL_NAME)

        # Load state dict
        # Use map_location to handle potential device mismatches safely
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        # Generate predictions
        seed_preds = inference_fn(model, test_loader, device)
        final_preds += seed_preds
        models_found += 1

        # Clean up to save memory
        del model, state_dict
        torch.cuda.empty_cache()

    if models_found == 0:
        logger.error("No trained models found. Cannot generate submission.")
        return

    # Compute average
    avg_preds = final_preds / models_found

    # ==========================================
    # 4. Save Submission
    # ==========================================
    # Load the sample submission to ensure correct format (columns: Insult, Date, Comment)
    if os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
        submission_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    else:
        # Fallback if sample submission is missing: use test df structure
        logger.warning(
            f"Sample submission not found at {Config.SAMPLE_SUBMISSION_PATH}. Constructing from test dataframe."
        )
        submission_df = df_test.copy()
        # Ensure Insult column exists and is at the start if possible
        if "Insult" not in submission_df.columns:
            submission_df.insert(0, "Insult", 0.0)

    # Verify lengths
    if len(submission_df) != len(avg_preds):
        logger.error(
            f"Length mismatch: Submission DF ({len(submission_df)}) vs Predictions ({len(avg_preds)})"
        )
        # Proceeding with assignment assuming index alignment as per metadata generation

    # Assign predictions
    submission_df["Insult"] = avg_preds

    # Save to submission directory
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    logger.info(f"Submission saved successfully to {submission_path}")
    logger.info(
        f"Prediction Stats: Mean={avg_preds.mean():.4f}, Min={avg_preds.min():.4f}, Max={avg_preds.max():.4f}"
    )
