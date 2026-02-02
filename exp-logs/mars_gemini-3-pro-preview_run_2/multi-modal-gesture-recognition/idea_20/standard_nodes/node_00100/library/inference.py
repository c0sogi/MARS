import os
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from itertools import groupby
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_device, load_checkpoint, ensure_dir
from library.model import DSG_CRCN
from library.data_loader import GestureDataset, collate_fn


class Predictor:
    """
    Inference engine for the Dual-Scale Supervised Gated-Cascaded Recurrent-Convolutional Network.
    Handles model loading, batch prediction, post-processing, and submission generation.
    """

    def __init__(self, checkpoint_path=None, batch_size=Config.BATCH_SIZE, device=None):
        """
        Initialize the Predictor.

        Args:
            checkpoint_path (str, optional): Path to the model checkpoint.
                                             Defaults to checking submission/ then cache/ directories.
            batch_size (int): Batch size for inference.
            device (torch.device, optional): Device to run inference on.
        """
        self.device = device if device else get_device()
        self.batch_size = batch_size

        # Initialize Model
        self.model = DSG_CRCN().to(self.device)

        # Determine checkpoint path
        if checkpoint_path is None:
            # Priority: Submission directory -> Cache directory
            submission_ckpt = os.path.join(Config.SUBMISSION_DIR, "best_model.pth")
            cache_ckpt = os.path.join(Config.CACHE_DIR, "best_model.pth")

            if os.path.exists(submission_ckpt):
                checkpoint_path = submission_ckpt
            elif os.path.exists(cache_ckpt):
                checkpoint_path = cache_ckpt
            else:
                print(
                    "Warning: No checkpoint found. Model will use random initialization."
                )
                checkpoint_path = None

        # Load weights if checkpoint exists
        if checkpoint_path:
            load_checkpoint(checkpoint_path, self.model, device=self.device)

        self.model.eval()

    def _post_process(self, probs):
        """
        Applies post-processing to frame-wise probabilities:
        1. Argmax to get discrete labels.
        2. Median Filtering to smooth noise.
        3. Decoding (collapsing repeats and removing background).

        Args:
            probs (np.ndarray): (T, NumClasses) array of probabilities.

        Returns:
            list: Ordered list of predicted gesture IDs (integers).
        """
        # 1. Discretize: (T, C) -> (T,)
        labels = np.argmax(probs, axis=1)

        # 2. Smooth: Median Filter with Nearest Neighbor Padding
        # This removes short spurious spikes while preserving boundaries
        labels_smoothed = median_filter(
            labels, size=Config.MEDIAN_FILTER_KERNEL, mode="nearest"
        )

        # 3. Decode: Collapse consecutive duplicates
        # e.g., [0, 0, 1, 1, 1, 0, 2] -> [0, 1, 0, 2]
        collapsed = [k for k, g in groupby(labels_smoothed)]

        # Remove background class (Index 0)
        # e.g., [0, 1, 0, 2] -> [1, 2]
        gestures = [g for g in collapsed if g != 0]

        return gestures

    def predict_dataset(self, dataset_split="test", limit=None):
        """
        Runs inference on the specified dataset split.

        Args:
            dataset_split (str): 'test', 'val', or 'train'.
            limit (int, optional): Limit the number of samples for debugging.

        Returns:
            tuple: (sample_ids, all_predictions)
                   sample_ids: List of sample ID strings.
                   all_predictions: List of lists containing predicted gesture IDs.
        """
        # Load Dataset
        # Uses the caching mechanism implemented in GestureDataset
        dataset = GestureDataset(dataset_split, load_cached_data=True, limit=limit)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Must be False to preserve order matching metadata
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        all_predictions = []

        print(f"Running inference on {len(dataset)} samples...")

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)

                # Forward Pass
                outputs = self.model(features, mask)

                # Extract Stage 3 Class Probabilities
                # Shape: (B, T, NumClasses)
                probs_batch = outputs["stage3_cls"]

                # Move to CPU for post-processing
                probs_batch_np = probs_batch.cpu().numpy()
                mask_np = mask.cpu().numpy()

                # Process each sequence in the batch
                for i in range(features.size(0)):
                    # Determine valid sequence length from mask
                    valid_length = int(mask_np[i].sum())

                    # Slice valid frames
                    sample_probs = probs_batch_np[i, :valid_length, :]

                    # Apply Post-Processing
                    predicted_gestures = self._post_process(sample_probs)
                    all_predictions.append(predicted_gestures)

        # Retrieve Sample IDs from metadata to ensure alignment
        # The DataLoader with shuffle=False preserves the order of the dataset
        sample_ids = dataset.metadata["sample_id"].tolist()

        if len(sample_ids) != len(all_predictions):
            print(
                f"Warning: Mismatch between samples ({len(sample_ids)}) and predictions ({len(all_predictions)})."
            )

        return sample_ids, all_predictions

    def generate_submission(self, output_file="submission.csv"):
        """
        Generates the submission file for the test set.

        Args:
            output_file (str): Name of the output CSV file.
        """
        sample_ids, predictions = self.predict_dataset("test")

        ensure_dir(Config.SUBMISSION_DIR)
        output_path = os.path.join(Config.SUBMISSION_DIR, output_file)

        print(f"Writing submission to {output_path}...")

        with open(output_path, "w") as f:
            for sid, preds in zip(sample_ids, predictions):
                # Format: SessionID,Label1,Label2,Label3
                # e.g., Session00001,2,12,3
                label_str = ",".join(map(str, preds))

                # Handle case with no predictions (write just ID or ID,)
                if label_str:
                    line = f"{sid},{label_str}\n"
                else:
                    line = f"{sid}\n"

                f.write(line)

        print("Submission generated successfully.")


def run_inference():
    """
    Helper function to instantiate the predictor and run the full inference pipeline.
    """
    predictor = Predictor()
    predictor.generate_submission()
