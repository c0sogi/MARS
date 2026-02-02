import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from library.config import WORKING_DIR, TEST_METADATA_PATH, BATCH_SIZE, SUBMISSION_DIR
from library.model import GestureGRU
from library.data_loader import GestureDataset, collate_fn
from library.utils import decode_predictions, save_submission


class Predictor:
    """
    Predictor class for running inference on the test set.
    Manages model loading, data loading, prediction loop, and submission generation.
    """

    def __init__(self, model_path, device=None):
        """
        Args:
            model_path (str): Path to the trained model checkpoint (.pth).
            device (torch.device, optional): Device to run inference on.
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = GestureGRU().to(self.device)

        if os.path.exists(model_path):
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded successfully from {model_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

        self.model.eval()

    def predict_loader(self, data_loader):
        """
        Runs inference on a DataLoader and returns a dictionary of predictions.

        Args:
            data_loader (DataLoader): Loader providing test data.

        Returns:
            dict: Mapping from sample_id (str) to list of gesture IDs (int).
        """
        predictions = {}

        with torch.no_grad():
            for features, _, lengths, ids in data_loader:
                features = features.to(self.device)
                lengths = lengths.to(self.device)

                # Forward pass
                # logits shape: (Batch, Time, NumClasses)
                logits = self.model(features, lengths)

                # Move to CPU for processing
                logits_np = logits.cpu().numpy()
                lengths_np = lengths.cpu().numpy()

                # Process each sample in the batch
                for i, sample_id in enumerate(ids):
                    length = lengths_np[i]
                    # Extract valid frames for this sequence (ignore padding)
                    valid_logits = logits_np[i, :length, :]

                    # Decode sequence using utility function (Argmax -> Median Filter -> RLE)
                    pred_seq = decode_predictions(valid_logits)
                    predictions[sample_id] = pred_seq

        return predictions

    def run_inference(
        self,
        test_metadata_path=TEST_METADATA_PATH,
        output_filename="submission.csv",
        batch_size=BATCH_SIZE,
        load_cached_data=True,
    ):
        """
        Orchestrates the full inference pipeline: loads data, runs model, saves submission.

        Args:
            test_metadata_path (str): Path to test metadata CSV.
            output_filename (str): Name of the output CSV file.
            batch_size (int): Batch size for inference.
            load_cached_data (bool): Whether to use cached pre-processed data.
        """
        # 1. Prepare Dataset
        # GestureDataset handles caching internally based on load_cached_data flag
        test_dataset = GestureDataset(
            metadata_path=test_metadata_path,
            load_cached_data=load_cached_data,
            mode="test",
        )

        if len(test_dataset) == 0:
            print("Warning: Test dataset is empty.")
            return {}

        # 2. Prepare DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        print(f"Starting inference on {len(test_dataset)} samples...")

        # 3. Predict
        predictions = self.predict_loader(test_loader)

        # 4. Save Submission
        save_submission(predictions, filename=output_filename)

        return predictions
