import os
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from itertools import groupby

from library.config import PATHS, get_hyperparams
from library.utils import set_seed, load_checkpoint
from library.model import MG_CRGN
from library.data_loader import get_dataloaders


class Predictor:
    """
    Predictor class for generating submissions using the trained MG-CRGN model.
    Handles model loading, inference, post-processing (smoothing), and CSV generation.
    """

    def __init__(self):
        self.hp = get_hyperparams()
        self.device = torch.device(self.hp["device"])
        set_seed(self.hp["seed"])

        # Initialize Model
        self.model = MG_CRGN().to(self.device)

        # Load Checkpoint
        self._load_best_model()

    def _load_best_model(self):
        """
        Loads the best model weights from the checkpoint directory.
        """
        checkpoint_path = PATHS["model_save_path"]
        try:
            epoch, val_loss = load_checkpoint(
                self.model,
                optimizer=None,
                path=checkpoint_path,
                device=self.hp["device"],
            )
            print(f"Loaded model from epoch {epoch} with Val Loss: {val_loss:.6f}")
        except FileNotFoundError:
            print(
                f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights."
            )

    def _decode_predictions(self, probs, mask):
        """
        Decodes frame-wise probabilities into gesture sequences.
        Applies Argmax -> Median Filter -> Collapse Duplicates -> Remove Background.

        Args:
            probs (torch.Tensor): (B, NumClasses, T) Class probabilities.
            mask (torch.Tensor): (B, T) Valid frame mask.

        Returns:
            list[list[int]]: List of predicted gesture IDs for each sample in batch.
        """
        # Convert to numpy
        probs_np = probs.detach().cpu().numpy()  # (B, C, T)
        mask_np = mask.detach().cpu().numpy()  # (B, T)

        predictions = []

        # Median filter size (approx 15 frames ~ 1.5 sec based on analysis)
        filter_size = 15

        for i in range(probs_np.shape[0]):
            # Get valid length
            valid_len = int(mask_np[i].sum())
            if valid_len == 0:
                predictions.append([])
                continue

            # Slice valid frames: (C, T_valid) -> (T_valid, C)
            # We only care about the first 21 channels (Classes 0-20)
            sample_probs = probs_np[i, :21, :valid_len].transpose(1, 0)

            # Argmax to get discrete labels
            raw_labels = np.argmax(sample_probs, axis=1)

            # Label-Space Smoothing: Median Filtering
            # mode='nearest' corresponds to "Nearest-Neighbor Padding" for boundary protection
            smooth_labels = median_filter(raw_labels, size=filter_size, mode="nearest")

            # Collapse repetitions and remove background (0)
            gesture_seq = []
            for key, group in groupby(smooth_labels):
                if key != 0:  # 0 is background
                    gesture_seq.append(int(key))

            predictions.append(gesture_seq)

        return predictions

    def generate_submission(self, load_cached_data=True):
        """
        Runs inference on the test set and generates the submission CSV.

        Args:
            load_cached_data (bool): Whether to use cached preprocessed data.
        """
        print("Initializing Test Dataloader...")
        # We only need the test loader (index 2)
        _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

        self.model.eval()
        results = []

        print("Starting Inference...")
        with torch.no_grad():
            for batch_idx, (features, _, mask, sample_ids) in enumerate(test_loader):
                features = features.to(self.device)
                mask = mask.to(self.device)

                # Forward Pass
                # Model returns list [out1, out2, out3]
                # We use the final stage output (Stage 3)
                outputs = self.model(features, mask)
                stage3_out = outputs[2]

                # Extract Class Probabilities (first 21 channels)
                # Shape: (B, NumClasses+1, T) -> (B, 21, T)
                cls_probs = stage3_out[:, :21, :]

                # Decode
                batch_preds = self._decode_predictions(cls_probs, mask)

                # Store results
                for sid, pred_seq in zip(sample_ids, batch_preds):
                    # Join labels with spaces for CSV format (single column)
                    pred_str = " ".join(map(str, pred_seq))
                    results.append((sid, pred_str))

        # Sort results by SessionID to ensure order (optional but good practice)
        results.sort(key=lambda x: x[0])

        # Write to CSV
        output_path = PATHS["submission"]
        print(f"Writing submission to {output_path}...")

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write("Id,Sequence\n")
            for sid, pred_str in results:
                # Extract integer ID from string (e.g., "Sample00300" -> 300)
                try:
                    sid_int = int("".join(filter(str.isdigit, sid)))
                except ValueError:
                    sid_int = sid

                # Format: Id,Sequence
                line = f"{sid_int},{pred_str}\n"
                f.write(line)

        print("Submission generation complete.")


def run_inference(load_cached_data=True):
    """
    Entry point for running the inference pipeline.
    """
    predictor = Predictor()
    predictor.generate_submission(load_cached_data=load_cached_data)
