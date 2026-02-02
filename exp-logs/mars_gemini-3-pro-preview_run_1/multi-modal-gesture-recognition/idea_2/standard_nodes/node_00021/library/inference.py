import os
import torch
import numpy as np
import pandas as pd
import scipy.ndimage
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import get_device, set_seed, ensure_dir
from library.model import GestureModel
from library.data_loader import GestureDataset, collate_fn


class InferenceManager:
    """
    Manages the inference process: model loading, prediction, post-processing,
    and submission file generation.
    """

    def __init__(self):
        self.device = get_device()
        self.model = self._load_model()

    def _load_model(self):
        """
        Initializes the model architecture and loads the best saved weights.
        """
        model = GestureModel()
        if os.path.exists(Config.MODEL_SAVE_PATH):
            # Load weights
            state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"Model loaded from {Config.MODEL_SAVE_PATH}")
        else:
            print(
                f"Warning: No model found at {Config.MODEL_SAVE_PATH}. Using random weights."
            )

        model.to(self.device)
        model.eval()
        return model

    def _median_filter(self, prediction_sequence):
        """
        Applies a temporal median filter to smooth predictions.

        Args:
            prediction_sequence (np.ndarray): 1D array of class indices.

        Returns:
            np.ndarray: Smoothed sequence.
        """
        return scipy.ndimage.median_filter(
            prediction_sequence, size=Config.MEDIAN_FILTER_KERNEL, mode="nearest"
        )

    def _decode_sequence(self, prediction_sequence):
        """
        Converts frame-wise predictions into an ordered list of gesture IDs.
        Filters out background class (0) and short segments.

        Args:
            prediction_sequence (np.ndarray): 1D array of class indices.

        Returns:
            list: Ordered list of recognized gesture IDs.
        """
        if len(prediction_sequence) == 0:
            return []

        gestures = []
        current_label = prediction_sequence[0]
        current_len = 1

        # Iterate starting from the second frame
        for i in range(1, len(prediction_sequence)):
            label = prediction_sequence[i]
            if label == current_label:
                current_len += 1
            else:
                # End of a segment
                if current_label != 0 and current_len >= Config.MIN_GESTURE_LENGTH:
                    gestures.append(int(current_label))

                current_label = label
                current_len = 1

        # Handle the last segment
        if current_label != 0 and current_len >= Config.MIN_GESTURE_LENGTH:
            gestures.append(int(current_label))

        return gestures

    def predict_all(self, limit=None, load_cached_data=True):
        """
        Runs inference on the test set and generates the submission file.

        Args:
            limit (int, optional): Limit number of samples for debugging.
            load_cached_data (bool): Flag required by prompt, though caching is
                                     handled internally by GestureDataset.
        """
        set_seed(Config.SEED)

        # Initialize Dataset and Loader
        # Note: GestureDataset handles the caching logic internally based on file existence.
        test_dataset = GestureDataset(split="test", augment=False, limit=limit)

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        results = []

        print("Starting inference on test set...")

        with torch.no_grad():
            # We need to track sample IDs. The dataset is ordered, so we can retrieve them by index
            # or we can modify the collate/dataset to return IDs.
            # Given the provided data_loader.py, the dataset returns (skel, audio, labels).
            # We will iterate the dataset indices to map back to sample_ids.

            global_idx = 0

            for batch_idx, (skeleton, audio, labels, lengths) in enumerate(test_loader):
                skeleton = skeleton.to(self.device)
                audio = audio.to(self.device)

                # Forward pass
                logits = self.model(skeleton, audio)

                # Get predictions: (B, T)
                predictions = torch.argmax(logits, dim=2).cpu().numpy()

                batch_size = skeleton.shape[0]

                for i in range(batch_size):
                    # Get sample ID
                    if global_idx < len(test_dataset.sample_ids):
                        sample_id = test_dataset.sample_ids[global_idx]
                    else:
                        # Should not happen
                        sample_id = f"Unknown_{global_idx}"

                    global_idx += 1

                    # Slice valid length (remove padding)
                    valid_len = lengths[i]
                    seq_pred = predictions[i, :valid_len]

                    # Post-processing
                    # 1. Median Filter
                    smoothed_pred = self._median_filter(seq_pred)

                    # 2. Decode Sequence
                    gesture_list = self._decode_sequence(smoothed_pred)

                    # Store result
                    results.append({"sample_id": sample_id, "gestures": gesture_list})

        self._save_submission(results)

    def _save_submission(self, results):
        """
        Formats and saves the results to a CSV file.
        Format: Id,Sequence (where Sequence is space-separated)
        """
        ensure_dir(Config.SUBMISSION_PATH)

        print(f"Saving submission to {Config.SUBMISSION_PATH}...")

        with open(Config.SUBMISSION_PATH, "w") as f:
            # Write Header
            f.write("Id,Sequence\n")

            for res in results:
                sample_id_str = res["sample_id"]
                gestures = res["gestures"]

                # Extract numeric ID (e.g., "Sample00300" -> 300)
                # Cite debug_lesson_3: Strictly Align Submission IDs with Grading Requirements
                try:
                    numeric_id = int("".join(filter(str.isdigit, sample_id_str)))
                except ValueError:
                    numeric_id = sample_id_str

                # Format Sequence: Space separated to avoid ragged CSV
                # Cite debug_lesson_2: Encapsulate Variable-Length Sequences to Prevent Ragged CSV Errors
                sequence_str = " ".join([str(g) for g in gestures])

                f.write(f"{numeric_id},{sequence_str}\n")

        print("Submission saved successfully.")
