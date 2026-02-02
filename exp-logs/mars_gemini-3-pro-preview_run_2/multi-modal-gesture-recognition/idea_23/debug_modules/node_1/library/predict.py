import os
import torch
import numpy as np
from library.config import Config
from library.model import SymG_CRCN
from library.dataset import get_dataloader
from library.utils import set_seed, apply_median_filter


class Predictor:
    """
    Handles the inference pipeline for the SymG-CRCN model.
    Loads the trained model, processes test data, applies post-processing,
    and generates the final submission file.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # Initialize model structure
        self.model = SymG_CRCN().to(self.device)

        # Load weights
        self._load_model()

    def _load_model(self):
        """Loads the best model checkpoint if available."""
        if os.path.exists(Config.BEST_MODEL_PATH):
            state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Loaded model from {Config.BEST_MODEL_PATH}")
        else:
            print(
                f"Warning: No checkpoint found at {Config.BEST_MODEL_PATH}. Model is untrained."
            )
            self.model.eval()

    def _nearest_neighbor_padding(self, seq):
        """
        Applies Nearest-Neighbor Padding to protect boundaries.
        Specifically, it fills single-frame background gaps (Class 0)
        that are sandwiched between identical valid gesture classes.

        Args:
            seq (np.array): Sequence of class indices.

        Returns:
            np.array: Processed sequence.
        """
        # Convert to list for mutable processing
        seq_list = list(seq)
        n = len(seq_list)

        if n < 3:
            return np.array(seq_list)

        # Iterate through sequence to find and fill gaps
        # Logic: If seq[i] is background (0) and neighbors are identical class C != 0, set seq[i] = C
        for i in range(1, n - 1):
            curr = seq_list[i]
            prev = seq_list[i - 1]
            next_val = seq_list[i + 1]

            if curr == Config.BACKGROUND_CLASS_ID:
                if prev == next_val and prev != Config.BACKGROUND_CLASS_ID:
                    seq_list[i] = prev

        return np.array(seq_list)

    def decode_sequence(self, logits, length):
        """
        Decodes frame-wise logits into a list of gesture IDs.

        Args:
            logits (torch.Tensor): Logits of shape (Time, NumClasses).
            length (int): Valid length of the sequence.

        Returns:
            list: Ordered list of gesture IDs (int).
        """
        # 1. Get discrete predictions
        # (T, C) -> (T,)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        # Truncate to valid sequence length
        valid_preds = preds[:length]

        # 2. Apply Median Filter Smoothing
        # Smooths out isolated noise spikes
        smoothed = apply_median_filter(valid_preds, kernel_size=Config.MEDIAN_FILTER_K)

        # 3. Apply Nearest Neighbor Padding
        # Protects boundaries by filling small gaps
        padded = self._nearest_neighbor_padding(smoothed)

        # 4. Collapse repeats and remove background
        collapsed = []
        prev = None
        for label in padded:
            if label != prev:
                if label != Config.BACKGROUND_CLASS_ID:
                    collapsed.append(int(label))
                prev = label

        return collapsed

    def predict(self, batch_size=Config.BATCH_SIZE):
        """
        Runs inference on the test set and saves the submission file.

        Args:
            batch_size (int): Batch size for inference.
        """
        print("Starting inference on test set...")

        # Get Test DataLoader
        # shuffle=False is required to maintain order for submission
        test_loader = get_dataloader(
            "test", batch_size=batch_size, shuffle=False, augment=False
        )

        results = []

        with torch.no_grad():
            for batch in test_loader:
                # Move inputs to device
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)

                # Lengths and IDs stay on CPU for post-processing
                lengths = batch["lengths"]
                sample_ids = batch["sample_ids"]

                # Forward Pass
                # Model expects lengths on device for packing
                outputs = self.model(features, mask, lengths.to(self.device))

                # Use Stage 3 (Sharpening) outputs for final prediction
                s3_logits = outputs["stage3_cls"]  # Shape: (B, T, C)

                # Process each sample in the batch
                for i in range(len(sample_ids)):
                    sid = sample_ids[i]
                    length = lengths[i].item()
                    seq_logits = s3_logits[i]  # Shape: (T, C)

                    # Decode sequence
                    pred_gestures = self.decode_sequence(seq_logits, length)

                    # Format string: "SessionID,g1,g2,g3"
                    pred_str = ",".join(map(str, pred_gestures))
                    results.append(f"{sid},{pred_str}")

        # Save results to file
        self._save_submission(results)

    def _save_submission(self, results):
        """Writes the results to the submission CSV file."""
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        with open(Config.SUBMISSION_PATH, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {Config.SUBMISSION_PATH}")
