import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import get_dataloaders
from library.model import SK_ARN


class Predictor:
    """
    Inference engine for the SK-ARN model.
    Handles model loading, sliding window inference, probability aggregation,
    and final sequence decoding.
    """

    def __init__(self, debug_max=None):
        """
        Initialize the Predictor.

        Args:
            debug_max (int, optional): Limit number of test samples for debugging.
        """
        self.device = get_device()
        set_seed(Config.SEED)

        # Load Data
        # We only need the test loader
        _, _, self.test_loader = get_dataloaders(debug_max=debug_max)

        # Initialize Model
        self.model = SK_ARN().to(self.device)
        self._load_checkpoint()
        self.model.eval()

    def _load_checkpoint(self):
        """Loads the best trained model weights."""
        if os.path.exists(Config.BEST_MODEL_PATH):
            # Load state dict
            state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model from {Config.BEST_MODEL_PATH}")
        else:
            raise FileNotFoundError(
                f"Model checkpoint not found at {Config.BEST_MODEL_PATH}"
            )

    def predict(self):
        """
        Runs inference on the test set.

        Returns:
            list: A list of tuples (sample_id, prediction_string).
        """
        # Store window predictions: sample_id -> list of (start_index, probabilities_tensor)
        window_preds = defaultdict(list)

        print(f"Starting inference on {len(self.test_loader.dataset)} windows...")

        with torch.no_grad():
            for batch_idx, (features, _, sample_ids, starts) in enumerate(
                self.test_loader
            ):
                features = features.to(self.device)

                # Forward pass
                outputs = self.model(features)

                # Get Stage 3 logits (Refined predictions)
                logits = outputs["stage3"]  # (B, T, C)

                # Convert to probabilities
                probs = F.softmax(logits, dim=2).cpu().numpy()  # (B, T, C)

                # Store predictions for aggregation
                for i, sample_id in enumerate(sample_ids):
                    start = starts[i].item()
                    window_prob = probs[i]  # (T, C)
                    window_preds[sample_id].append((start, window_prob))

        return self._aggregate_and_decode(window_preds)

    def _aggregate_and_decode(self, window_preds):
        """
        Aggregates overlapping window probabilities and decodes the sequence.

        Args:
            window_preds (dict): Dictionary mapping sample_id to list of (start, prob) tuples.

        Returns:
            list: List of (sample_id, prediction_string) tuples.
        """
        results = []

        print("Aggregating probabilities and decoding sequences...")

        for sample_id, windows in window_preds.items():
            # 1. Determine full sequence length
            max_len = 0
            for start, prob in windows:
                end = start + prob.shape[0]
                if end > max_len:
                    max_len = end

            # 2. Aggregate Probabilities (Temporal Ensembling)
            num_classes = Config.NUM_CLASSES
            accum_probs = np.zeros((max_len, num_classes), dtype=np.float32)
            counts = np.zeros((max_len, 1), dtype=np.float32)

            for start, prob in windows:
                end = start + prob.shape[0]
                # Handle edge case where window might extend beyond max_len if calculated differently
                # But here max_len is derived from windows, so it fits.
                accum_probs[start:end] += prob
                counts[start:end] += 1

            # Average probabilities
            avg_probs = accum_probs / np.maximum(counts, 1.0)

            # 3. Decode Sequence
            # Argmax
            pred_indices = np.argmax(avg_probs, axis=1)  # (T,)

            # Run-Length Encoding (RLE)
            decoded_sequence = []
            if len(pred_indices) > 0:
                # Collapse consecutive duplicates
                collapsed = [pred_indices[0]]
                for idx in range(1, len(pred_indices)):
                    if pred_indices[idx] != pred_indices[idx - 1]:
                        collapsed.append(pred_indices[idx])

                # Remove background class (0)
                decoded_sequence = [
                    x for x in collapsed if x != Config.BACKGROUND_CLASS_ID
                ]

            # Format as string: "Label1,Label2,..."
            prediction_str = ",".join(map(str, decoded_sequence))
            results.append((sample_id, prediction_str))

        return results

    def generate_submission(self):
        """
        Runs the full inference pipeline and saves the submission CSV.
        """
        results = self.predict()

        # Sort by sample_id for orderly output
        results.sort(key=lambda x: x[0])

        output_path = Config.SUBMISSION_FILE
        print(f"Saving submission to {output_path}...")

        # Write to CSV in the format: SessionID,Label1,Label2,...
        with open(output_path, "w") as f:
            for sample_id, pred_str in results:
                if pred_str:
                    line = f"{sample_id},{pred_str}"
                else:
                    # If no gestures detected, just the ID (or ID with empty string)
                    line = f"{sample_id}"
                f.write(line + "\n")

        print("Submission generated successfully.")
