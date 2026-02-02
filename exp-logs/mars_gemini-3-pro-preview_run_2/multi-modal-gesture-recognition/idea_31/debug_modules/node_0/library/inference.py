import os
import torch
import numpy as np
import scipy.ndimage
import pandas as pd
from library.config import (
    WORKING_DIR,
    SUBMISSION_DIR,
    BATCH_SIZE,
    SEED,
    BACKGROUND_CLASS_ID,
    MEDIAN_FILTER_KERNEL,
    NUM_CLASSES,
    INPUT_DIM,
    HIDDEN_DIM,
)
from library.utils import setup_logger, set_seed
from library.data_loader import get_dataloaders
from library.model import MSE_GCN

# Ensure reproducibility
set_seed(SEED)


class Predictor:
    """
    Handles inference for the MSE-GCN model.
    """

    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger("inference.log")

        # Initialize Model
        self.model = MSE_GCN().to(self.device)

        # Load Weights
        if model_path is None:
            model_path = os.path.join(WORKING_DIR, "checkpoints", "best_model.pth")

        if os.path.exists(model_path):
            self.logger.info(f"Loading model from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            self.logger.error(f"Model checkpoint not found at {model_path}")
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

        self.model.eval()

    def post_process_sequence(self, cls_indices, mask_len):
        """
        Applies Median Filter to discrete labels and decodes the sequence.

        Args:
            cls_indices (np.ndarray): 1D array of class indices for a single sample.
            mask_len (int): Valid length of the sequence.

        Returns:
            list: Decoded list of gesture IDs.
        """
        # Truncate to valid length
        valid_seq = cls_indices[:mask_len]

        if len(valid_seq) == 0:
            return []

        # Apply Median Filter to discrete labels
        # Mode 'nearest' corresponds to Nearest-Neighbor Padding for boundaries
        filtered_seq = scipy.ndimage.median_filter(
            valid_seq, size=MEDIAN_FILTER_KERNEL, mode="nearest"
        )

        # Decode: Collapse duplicates and remove background
        decoded_seq = []
        prev = -1

        for token in filtered_seq:
            if token != prev:
                if token != BACKGROUND_CLASS_ID:
                    decoded_seq.append(int(token))
                prev = token

        return decoded_seq

    def predict(self, test_loader):
        """
        Runs inference on the test set.

        Args:
            test_loader: DataLoader for the test set.

        Returns:
            dict: Mapping from sample_id to list of predicted gesture IDs.
        """
        self.logger.info("Starting inference...")
        predictions = {}

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                sample_ids = batch["sample_ids"]

                # Forward pass
                # Returns list of dicts for each stage. We use the last stage.
                stage_outputs = self.model(features, mask, lengths)
                final_stage = stage_outputs[-1]
                cls_probs = final_stage["cls"]  # (B, T, C)

                # Get discrete labels (Argmax)
                cls_preds = torch.argmax(cls_probs, dim=2).cpu().numpy()  # (B, T)
                lengths_np = lengths.cpu().numpy()

                # Process each sample in the batch
                for i, sample_id in enumerate(sample_ids):
                    seq_len = int(lengths_np[i])
                    raw_seq = cls_preds[i]

                    # Post-process
                    decoded_seq = self.post_process_sequence(raw_seq, seq_len)
                    predictions[sample_id] = decoded_seq

        self.logger.info(
            f"Inference complete. Generated predictions for {len(predictions)} samples."
        )
        return predictions


def generate_submission(load_cached_data=True):
    """
    Main function to generate the submission file.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    logger = setup_logger("inference.log")

    # 1. Get DataLoaders (only need test_loader)
    # This handles caching internally via process_dataset
    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, load_cached_data=load_cached_data
    )

    # 2. Initialize Predictor
    predictor = Predictor()

    # 3. Run Inference
    results = predictor.predict(test_loader)

    # 4. Write Submission File
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    logger.info(f"Writing submission to {submission_path}")

    try:
        with open(submission_path, "w") as f:
            # Iterate through sorted keys for deterministic output order
            for sample_id in sorted(results.keys()):
                pred_list = results[sample_id]
                # Format: SessionID,Label1,Label2,...
                # Example: Session00001,2,12,3
                pred_str = ",".join(map(str, pred_list))
                line = f"{sample_id},{pred_str}\n"
                f.write(line)

        logger.info("Submission file generated successfully.")

    except Exception as e:
        logger.error(f"Failed to write submission file: {e}")
        raise e
