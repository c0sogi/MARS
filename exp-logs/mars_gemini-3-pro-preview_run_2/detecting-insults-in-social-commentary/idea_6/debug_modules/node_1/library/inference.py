import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from library.config import Config
from library.utils import seed_everything, Logger
from library.data import load_and_process_data, InsultDataset
from library.model import InsultModel
from library.train import inference_fn


def predict(load_cached_data=True):
    """
    Executes the inference pipeline: loads data, loads trained models for each fold,
    generates predictions, averages them, and saves the submission file.

    Args:
        load_cached_data (bool): Whether to attempt loading pre-processed parquet files.
    """
    # 1. Setup
    seed_everything(Config.seed)
    Config.setup()

    # Initialize Logger
    log_path = os.path.join(Config.output_dir, "inference_log.txt")
    logger = Logger(log_path)
    logger.log("Starting Inference Pipeline...")

    device = Config.device
    logger.log(f"Device: {device}")

    # 2. Load Data
    # We only require the test dataframe
    _, _, df_test = load_and_process_data(load_cached_data=load_cached_data)
    logger.log(f"Test Data Size: {len(df_test)}")

    # 3. Prepare DataLoader
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    test_dataset = InsultDataset(df_test, tokenizer, Config.max_len, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 4. Inference Loop (Ensemble)
    avg_preds = np.zeros(len(df_test))
    models_found = 0

    for fold in range(Config.n_folds):
        model_path = os.path.join(Config.output_dir, f"model_fold_{fold}.pth")

        if not os.path.exists(model_path):
            logger.log(f"Model for fold {fold} not found at {model_path}. Skipping.")
            continue

        logger.log(f"Predicting with model fold {fold}...")

        # Initialize model structure
        model = InsultModel(pretrained=False)
        model.to(device)

        # Load trained weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        # Run inference
        fold_preds = inference_fn(model, test_loader, device)

        # Accumulate predictions
        avg_preds += fold_preds
        models_found += 1

        # Cleanup to free GPU memory
        del model, state_dict, fold_preds
        torch.cuda.empty_cache()
        gc.collect()

    # 5. Process Results
    if models_found == 0:
        logger.log("Error: No trained models found. Cannot generate predictions.")
        return

    # Average the probabilities
    avg_preds /= models_found

    # 6. Save Submission
    # Create submission dataframe matching the structure of the input/sample
    submission = df_test.copy()
    submission[Config.target_col] = avg_preds

    submission.to_csv(Config.submission_path, index=False)
    logger.log(f"Submission saved to {Config.submission_path}")
    logger.log("Inference Complete.")
