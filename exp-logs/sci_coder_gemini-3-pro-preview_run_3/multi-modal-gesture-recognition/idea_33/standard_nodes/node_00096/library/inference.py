import os
import torch
import torch.nn.functional as F
import numpy as np
from library.config import (
    WINDOW_SIZE,
    NUM_CLASSES,
    BATCH_SIZE,
    MODEL_SAVE_PATH,
    SUBMISSION_FILE,
    SEED,
)
from library.model import RGHCMN
from library.utils import filter_short_segments
from library.data_loader import get_data_loaders


class SequencePredictor:
    """
    Handles inference on full video sequences using a sliding window approach
    with temporal aggregation and deep supervision.
    """

    def __init__(self, model_path=MODEL_SAVE_PATH, output_file=SUBMISSION_FILE):
        self.model_path = model_path
        self.output_file = output_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        """
        Executes the full inference pipeline:
        1. Loads data and model.
        2. Runs sliding window inference.
        3. Aggregates predictions.
        4. Applies post-processing.
        5. Generates submission file.
        """
        # 1. Load Data
        print("Loading test data...")
        # We only need the test loader
        _, _, test_loader = get_data_loaders(batch_size=BATCH_SIZE)
        dataset = test_loader.dataset

        # 2. Load Model
        print(f"Loading model from {self.model_path}...")
        model = RGHCMN().to(self.device)

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # Handle both full checkpoint dicts and direct state_dicts
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(
                f"Warning: Model file {self.model_path} not found. Using random weights."
            )

        model.eval()

        # 3. Prepare Accumulators for Sequence Reconstruction
        # We need to reconstruct the full time-series for each sample from overlapping windows.
        # sample_probs: Stores sum of probabilities for each frame
        # sample_counts: Stores how many windows covered each frame (for averaging)
        sample_probs = {}
        sample_counts = {}

        # Initialize buffers on CPU to save GPU memory
        for i, sample_id in enumerate(dataset.sample_ids):
            start, end = dataset.sample_boundaries[i]
            length = end - start
            # Shape: (Length, NumClasses)
            sample_probs[i] = torch.zeros(
                (length, NUM_CLASSES), device="cpu", dtype=torch.float
            )
            sample_counts[i] = torch.zeros((length, 1), device="cpu", dtype=torch.float)

        # 4. Inference Loop
        print("Running sliding window inference...")
        with torch.no_grad():
            for batch_x, _, batch_indices, batch_starts in test_loader:
                batch_x = batch_x.to(self.device)

                # Forward pass
                outputs = model(batch_x)
                # Use Stage 3 logits for final prediction (Deep Supervision)
                logits = outputs["logits_3"]
                probs = F.softmax(logits, dim=2)  # (Batch, Time, Classes)

                # Move to CPU for accumulation
                probs = probs.cpu()

                # Distribute batch predictions to their respective sequences
                for k in range(len(batch_indices)):
                    s_idx = batch_indices[k].item()
                    r_start = batch_starts[k].item()

                    # Get total length of this specific sequence
                    total_len = sample_probs[s_idx].shape[0]

                    # Determine valid length for this window
                    # The dataset pads the last window if it's shorter than WINDOW_SIZE.
                    # We must only accumulate the valid part of the window that falls within the sequence.
                    valid_len = min(WINDOW_SIZE, total_len - r_start)

                    # Accumulate probabilities and counts
                    sample_probs[s_idx][r_start : r_start + valid_len] += probs[
                        k, :valid_len, :
                    ]
                    sample_counts[s_idx][r_start : r_start + valid_len] += 1.0

        # 5. Generate Predictions & Submission
        print("Generating final predictions...")
        results = []

        for i, sample_id in enumerate(dataset.sample_ids):
            # Average the probabilities
            counts = sample_counts[i]
            # Avoid division by zero (though all frames should be covered at least once)
            counts[counts == 0] = 1.0

            avg_probs = sample_probs[i] / counts

            # Get frame-wise labels
            frame_preds = torch.argmax(avg_probs, dim=1).numpy()

            # Post-processing: Filter short segments
            gesture_ids = filter_short_segments(frame_preds)

            # Format: SessionID,g1,g2,g3
            gestures_str = ",".join(map(str, gesture_ids))

            # Construct line. If no gestures, it will be "SessionID" (or "SessionID," depending on handling)
            # Based on requirements, we output "SessionID,2,12,3"
            if gestures_str:
                line = f"{sample_id},{gestures_str}"
            else:
                line = f"{sample_id}"

            results.append(line)

        # 6. Save to CSV
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {self.output_file}")
