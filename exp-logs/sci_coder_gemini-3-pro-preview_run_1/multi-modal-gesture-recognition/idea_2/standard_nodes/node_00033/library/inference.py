import os
import torch
import numpy as np
import pandas as pd
import scipy.ndimage
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import get_device, set_seed, ensure_dir
from library.model import MultiStreamGRU
from library.data_loader import GestureDataset, collate_fn
from library.metrics import decode_sequence


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
        model = MultiStreamGRU()
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

            for batch_idx, (skeleton, audio, labels, lengths, indices) in enumerate(
                test_loader
            ):
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

                    # Post-processing & Decode (using shared metrics logic)
                    gesture_list = decode_sequence(seq_pred, apply_median_filter=True)

                    # Store result
                    results.append({"sample_id": sample_id, "gestures": gesture_list})

        self._save_submission(results)

    def _save_submission(self, results):
        """
        Formats and saves the results to a CSV file.
        Format: Id,Sequence
        Where Id is integer (300) and Sequence is space-separated integers (2 12 3).
        """
        ensure_dir(Config.SUBMISSION_PATH)

        print(f"Saving submission to {Config.SUBMISSION_PATH}...")

        with open(Config.SUBMISSION_PATH, "w") as f:
            # Write Header
            f.write("Id,Sequence\n")

            for res in results:
                sample_id_str = res["sample_id"]
                gestures = res["gestures"]

                # Convert Sample00300 -> 300
                # Cite debug_lesson_3: Strictly Align Submission IDs with Grading Requirements
                try:
                    # Remove non-numeric characters to extract ID
                    sid = int("".join(filter(str.isdigit, sample_id_str)))
                except ValueError:
                    # Fallback if format is unexpected
                    sid = sample_id_str

                # Format sequence as space-separated string
                # Cite debug_lesson_2: Encapsulate Variable-Length Sequences to Prevent Ragged CSV Errors
                sequence_str = " ".join([str(g) for g in gestures])

                # Write row: 300,2 12 3
                f.write(f"{sid},{sequence_str}\n")

        print("Submission saved successfully.")
