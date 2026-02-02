import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.model import ICRCN
from library.data_loader import get_loaders
from library.utils import apply_median_filter, decode_predictions


class Predictor:
    """
    Handles inference for the IC-RCN model on the test dataset.
    """

    def __init__(self, model_path=None, device=None):
        self.device = device if device else torch.device(Config.DEVICE)
        self.model = ICRCN().to(self.device)

        # Load checkpoint
        path = model_path if model_path else Config.BEST_MODEL_PATH
        if os.path.exists(path):
            print(f"Loading model from {path}...")
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Checkpoint not found at {path}. Using random initialization."
            )

        self.model.eval()

    def predict(self, test_loader):
        """
        Runs inference on the test loader.

        Args:
            test_loader (DataLoader): DataLoader for the test set.

        Returns:
            list: A list of predicted gesture sequences (lists of ints).
        """
        all_predictions = []

        with torch.no_grad():
            for features, _, lengths in test_loader:
                features = features.to(self.device)

                # Forward pass
                outputs = self.model(features)

                # Use Refinement Stage 2 outputs for final prediction
                logits = outputs["ref2"]  # (B, C, T)
                probs = torch.softmax(logits, dim=1)

                # Move to CPU for post-processing
                probs_np = probs.cpu().numpy()
                lengths_np = lengths.cpu().numpy()

                # Process each sample in the batch
                for i in range(len(features)):
                    length = lengths_np[i]

                    # Extract valid frames (remove padding)
                    # Shape: (C, T) -> (T, C)
                    sample_probs = probs_np[i, :, :length].transpose(1, 0)

                    # 1. Apply Median Filter Smoothing
                    # library.utils.apply_median_filter uses mode='edge' (nearest neighbor padding)
                    smoothed_preds = apply_median_filter(sample_probs, kernel_size=5)

                    # 2. Decode to Sequence (collapse repetitions, remove background)
                    pred_seq = decode_predictions(smoothed_preds)

                    all_predictions.append(pred_seq)

        return all_predictions


def generate_submission(load_cached_data=True):
    """
    Main function to generate the submission file.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 1. Get Data Loaders
    # We only need test_loader and test_ids
    print("Loading test data...")
    _, _, test_loader, test_ids = get_loaders(load_cached_data=load_cached_data)

    # 2. Initialize Predictor
    predictor = Predictor()

    # 3. Run Inference
    print("Running inference...")
    predictions = predictor.predict(test_loader)

    # 4. Format Submission
    # Ensure alignment between ids and predictions
    if len(predictions) != len(test_ids):
        raise ValueError(
            f"Mismatch between predictions ({len(predictions)}) and test IDs ({len(test_ids)})"
        )

    submission_lines = []
    for sample_id, pred_seq in zip(test_ids, predictions):
        # Format: SessionID,Label1,Label2,...
        # Join labels with commas
        labels_str = ",".join(map(str, pred_seq))

        if labels_str:
            line = f"{sample_id},{labels_str}"
        else:
            # Handle case with no detected gestures
            line = f"{sample_id},"

        submission_lines.append(line)

    # 5. Save to File
    output_path = Config.SUBMISSION_FILE
    with open(output_path, "w") as f:
        for line in submission_lines:
            f.write(line + "\n")

    print(f"Submission saved to {output_path}")
