import os
import csv
import torch
import numpy as np
import pandas as pd
from itertools import groupby
from library.config import Config
from library.utils import setup_logger, set_seed
from library.model import VIARN
from library.data_processing import DataProcessor


class InferenceEngine:
    """
    Manages the inference process for the VI-ARN model.
    Handles data loading, sliding window prediction, decoding, and submission generation.
    """

    def __init__(self):
        self.logger = setup_logger(name="Inference")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Model
        self.model = VIARN().to(self.device)

        # Load Weights
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.logger.info(f"Loading model weights from {Config.MODEL_SAVE_PATH}")
            state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            self.logger.error(f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}")
            raise FileNotFoundError("Model checkpoint missing.")

        self.model.eval()
        self.processor = DataProcessor()

    def predict_sequence(self, features):
        """
        Performs inference on a single sequence using sliding windows.

        Args:
            features (np.ndarray): Shape (T, InputDim)

        Returns:
            probs (np.ndarray): Shape (T, NumClasses) - Frame-wise probabilities
        """
        seq_len = features.shape[0]
        window_size = Config.WINDOW_SIZE
        stride = Config.STRIDE

        # Prepare buffers for aggregation
        # We accumulate probabilities and counts for averaging overlapping windows
        probs_sum = np.zeros((seq_len, Config.NUM_CLASSES), dtype=np.float32)
        counts = np.zeros((seq_len, 1), dtype=np.float32)

        # Generate windows
        windows = []
        indices = []

        # Handle short sequences (pad if necessary)
        if seq_len < window_size:
            pad_len = window_size - seq_len
            # Pad features with edge values
            padded_feat = np.pad(features, ((0, pad_len), (0, 0)), mode="edge")
            windows.append(padded_feat)
            indices.append((0, seq_len))  # Valid range
        else:
            # Sliding window
            curr = 0
            while curr + window_size <= seq_len:
                windows.append(features[curr : curr + window_size])
                indices.append((curr, curr + window_size))
                curr += stride

            # Handle last window if it doesn't cover the end perfectly
            if indices[-1][1] < seq_len:
                start = seq_len - window_size
                windows.append(features[start:seq_len])
                indices.append((start, seq_len))

        if not windows:
            return np.zeros((seq_len, Config.NUM_CLASSES))

        # Batch inference
        # Convert to tensor
        batch_windows = np.array(windows)  # (B, WindowSize, InputDim)
        batch_tensor = torch.from_numpy(batch_windows).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_tensor)
            # Use Stage 3 output for final prediction
            batch_probs = outputs["stage3"].cpu().numpy()  # (B, WindowSize, NumClasses)

        # Aggregate
        for i, (start, end) in enumerate(indices):
            # If sequence was padded (short sequence case)
            if seq_len < window_size:
                valid_len = end - start
                probs_sum[start:end] += batch_probs[i, :valid_len]
                counts[start:end] += 1
            else:
                # Standard sliding window
                # Note: indices are absolute for the sequence
                # batch_probs[i] corresponds to windows[i]

                # Check if this is the special last window case that might overlap weirdly
                # The window corresponds to features[start:end]
                probs_sum[start:end] += batch_probs[i]
                counts[start:end] += 1

        # Avoid division by zero (should not happen with correct logic)
        counts[counts == 0] = 1.0

        avg_probs = probs_sum / counts
        return avg_probs

    def decode_predictions(self, probs):
        """
        Decodes frame-wise probabilities into a list of gesture IDs.
        Applies argmax and Run-Length Encoding (RLE), ignoring background (0).

        Args:
            probs (np.ndarray): (T, NumClasses)

        Returns:
            List[int]: Ordered list of recognized gesture IDs.
        """
        # Argmax to get class indices
        preds = np.argmax(probs, axis=1)

        # Run-Length Encoding to collapse duplicates
        # groupby returns consecutive keys
        collapsed = [k for k, g in groupby(preds)]

        # Filter out background class (0)
        gestures = [k for k in collapsed if k != 0]

        return gestures

    def generate_submission(self, debug_subset=None):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        self.logger.info("Starting submission generation...")

        # 1. Load Test Metadata to get Sample IDs in order
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)
        if debug_subset:
            test_df = test_df.head(debug_subset)

        sample_ids = test_df["sample_id"].tolist()

        # 2. Process Test Data (uses caching)
        # This returns concatenated features and boundaries
        features, _, boundaries = self.processor.process_dataset(
            Config.TEST_METADATA_PATH,
            Config.CACHE_TEST_PATH,
            is_train=False,
            debug_size=debug_subset,
        )

        if len(features) == 0:
            self.logger.error("No test features processed.")
            return

        results = []

        # 3. Iterate and Predict
        # boundaries has N+1 elements.
        # sample_ids has N elements.

        for i, sample_id in enumerate(sample_ids):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            seq_features = features[start_idx:end_idx]

            if seq_features.shape[0] == 0:
                self.logger.warning(f"Empty features for {sample_id}")
                predicted_gestures = []
            else:
                # Predict
                seq_probs = self.predict_sequence(seq_features)

                # Decode
                predicted_gestures = self.decode_predictions(seq_probs)

            # Format: SessionID,Label1,Label2,...
            # Convert list to string
            if predicted_gestures:
                labels_str = ",".join(map(str, predicted_gestures))
                row_str = f"{sample_id},{labels_str}"
            else:
                # If no gestures detected, just ID
                row_str = f"{sample_id}"

            results.append(row_str.split(","))

        # 4. Write to CSV
        output_path = Config.SUBMISSION_PATH
        self.logger.info(f"Writing submission to {output_path}")

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            # No header required by the prompt example, but standard CSV usually has one.
            # The prompt example: "Session00001,2,12,3".
            # It does not explicitly forbid a header, but usually challenges specify if one is needed.
            # The prompt says: "For instance: Session00001,2,12,3".
            # I will write rows directly.
            writer.writerows(results)

        self.logger.info("Submission generation complete.")


def run_inference(debug_subset=None):
    """
    Entry point function to run the inference pipeline.
    """
    set_seed(Config.SEED)
    engine = InferenceEngine()
    engine.generate_submission(debug_subset=debug_subset)
