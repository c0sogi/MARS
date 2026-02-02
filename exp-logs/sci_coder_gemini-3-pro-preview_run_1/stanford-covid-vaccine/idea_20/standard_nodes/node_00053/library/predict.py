import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import (
    get_logger,
    load_checkpoint,
    generate_submission_file,
    seed_everything,
)
from library.data import get_dataloaders
from library.model import Net


class Predictor:
    """
    Handles model inference on the test dataset.
    """

    def __init__(self, model_path, device, logger=None):
        """
        Args:
            model_path (str): Path to the model checkpoint.
            device (str): Device to run inference on ('cpu' or 'cuda').
            logger (logging.Logger, optional): Logger instance.
        """
        self.model_path = model_path
        self.device = device
        self.logger = logger or get_logger("Predictor")
        self.model = Net().to(self.device)
        self._load_model()

    def _load_model(self):
        """Loads the model weights from the checkpoint."""
        self.logger.info(f"Loading model checkpoint from {self.model_path}...")
        load_checkpoint(self.model_path, self.model, device=self.device)
        self.model.eval()

    def predict(self, data_loader):
        """
        Runs inference on the provided data loader.

        Args:
            data_loader (DataLoader): The test data loader.

        Returns:
            np.ndarray: Predictions of shape (N_samples, Seq_Len, n_targets).
        """
        self.logger.info("Starting inference on test set...")
        all_preds = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Move inputs to device
                sequence = batch["sequence"].to(self.device)
                loop_type = batch["loop_type"].to(self.device)
                distance = batch["distance"].to(self.device)

                # Forward pass
                # Output shape: (Batch, Seq_Len, 3)
                outputs = self.model(sequence, loop_type, distance)

                # Move to CPU and collect
                all_preds.append(outputs.cpu().numpy())

        # Concatenate all batches: (N_samples, Seq_Len, 3)
        final_predictions = np.concatenate(all_preds, axis=0)
        self.logger.info(f"Inference complete. Output shape: {final_predictions.shape}")
        return final_predictions


def run_prediction(debug=False, load_cached_data=True):
    """
    Orchestrates the prediction pipeline: loading data, running inference, and saving submission.

    Args:
        debug (bool): If True, runs in debug mode (subset of data).
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
    """
    # 1. Setup
    if debug:
        Config.debug = True

    seed_everything(Config.seed)
    logger = get_logger(
        "Predict", log_file=os.path.join(Config.working_dir, "predict.log")
    )

    # 2. Data Loading
    logger.info("Initializing data loaders...")
    # We only need the test_loader. The function returns (train, val, test).
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Model Inference
    predictor = Predictor(
        model_path=Config.model_save_path, device=Config.device, logger=logger
    )
    predictions = predictor.predict(test_loader)

    # 4. Submission Generation
    logger.info("Reading test metadata for submission generation...")
    # We need the IDs and Sequences to construct the submission file
    df_test = pd.read_parquet(Config.test_file)

    # If debugging, ensure we only take the relevant subset of metadata
    if Config.debug:
        df_test = df_test.head(Config.debug_subset_size)

    ids = df_test["id"].tolist()
    sequences = df_test["sequence"].tolist()

    # Validate lengths match
    if len(ids) != len(predictions):
        logger.error(f"Mismatch: {len(ids)} IDs vs {len(predictions)} predictions.")
        raise ValueError("Number of test samples does not match number of predictions.")

    logger.info(f"Generating submission file at {Config.submission_path}...")
    generate_submission_file(
        ids=ids,
        sequences=sequences,
        predictions=predictions,
        output_path=Config.submission_path,
    )

    logger.info("Prediction pipeline finished successfully.")
