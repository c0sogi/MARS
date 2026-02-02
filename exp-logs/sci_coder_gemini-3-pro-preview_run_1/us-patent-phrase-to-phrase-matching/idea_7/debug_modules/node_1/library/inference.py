import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from library.config import Config
from library.utils import get_logger
from library.data import prepare_data, PhraseDataset
from library.model import DebertaV3Regressor

# Initialize logger for inference
logger = get_logger(os.path.join(Config.working_dir, "inference.log"))


def inference_fn(model, dataloader, device):
    """
    Generates predictions for a given model and dataloader.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        np.array: Array of predicted scores.
    """
    model.eval()
    preds = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            # Mixed Precision Inference
            with torch.cuda.amp.autocast(enabled=Config.use_fp16):
                outputs = model(input_ids, attention_mask, token_type_ids)

            # Move outputs to CPU and convert to numpy
            preds.append(outputs.cpu().numpy())

    # Concatenate all batch predictions
    predictions = np.concatenate(preds)
    return predictions


def run_inference():
    """
    Orchestrates the inference process:
    1. Loads the test data.
    2. Iterates through all trained fold models.
    3. Generates predictions and computes the ensemble average.
    4. Saves the submission file.
    """
    logger.info("Starting Inference...")

    # 1. Load Data
    # We use prepare_data to get the processed test dataframe (with context injected)
    _, test_df = prepare_data(load_cached_data=True)

    # Sort test_df by id to ensure alignment with sample_submission if needed,
    # though usually we just map by ID. We'll stick to the order in test_df
    # and construct the submission dataframe from it.

    # 2. Prepare DataLoader
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # is_train=False ensures the dataset doesn't look for 'score' column
    test_dataset = PhraseDataset(
        test_df, tokenizer, max_length=Config.max_length, is_train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size * 2,  # Larger batch size for inference
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Ensemble Inference
    # Initialize array to store sum of predictions
    avg_preds = np.zeros(len(test_df))

    device = Config.device
    models_found = 0

    for fold in range(Config.n_folds):
        model_path = os.path.join(Config.models_dir, f"model_fold_{fold}.pth")

        if not os.path.exists(model_path):
            logger.warning(
                f"Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        logger.info(f"Predicting with model fold {fold}...")

        # Load Model
        model = DebertaV3Regressor(Config.model_name, pretrained=False)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        # Predict
        fold_preds = inference_fn(model, test_loader, device)
        avg_preds += fold_preds
        models_found += 1

        # Cleanup to free memory
        del model, state_dict, fold_preds
        torch.cuda.empty_cache()
        gc.collect()

    if models_found == 0:
        raise RuntimeError("No trained models found in the models directory.")

    # Compute Average
    avg_preds /= models_found

    # Clip predictions to valid range [0, 1] as scores are bounded
    avg_preds = np.clip(avg_preds, 0, 1)

    # 4. Create Submission
    submission = pd.DataFrame({"id": test_df["id"], "score": avg_preds})

    # Ensure output directory exists
    os.makedirs(Config.submission_dir, exist_ok=True)

    # Save
    submission.to_csv(Config.submission_path, index=False)
    logger.info(f"Submission saved to {Config.submission_path}")
    logger.info(f"Head of submission:\n{submission.head()}")

    return submission
