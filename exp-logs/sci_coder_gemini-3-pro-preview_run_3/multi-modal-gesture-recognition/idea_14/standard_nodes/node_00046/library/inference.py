import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import RSKARN
from library.dataset import GestureDataset
from library.utils import run_length_encoding


class InferenceEngine:
    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path (str): Path to the trained model weights. Defaults to Config.MODEL_SAVE_PATH.
            device (str): Device to run inference on. Defaults to Config.DEVICE.
        """
        self.device = device if device else Config.DEVICE
        self.model_path = model_path if model_path else Config.MODEL_SAVE_PATH

        # Initialize Model
        self.model = RSKARN().to(self.device)
        self._load_model()
        self.model.eval()

    def _load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

    def predict_sliding_window(self, features):
        """
        Performs sliding window inference with temporal ensembling.
        Args:
            features: (T, InputDim) tensor
        Returns:
            probs: (T, NumClasses) tensor of averaged probabilities
        """
        T, D = features.shape
        window_size = Config.WINDOW_SIZE
        stride = Config.STRIDE

        # Prepare accumulator for probabilities
        probs_acc = torch.zeros((T, Config.NUM_CLASSES), device=self.device)
        count_acc = torch.zeros((T, 1), device=self.device)

        # Generate windows
        windows = []
        indices = []

        # Handle case where sequence is shorter than window
        if T < window_size:
            # Pad to window size
            pad_len = window_size - T
            feat_pad = F.pad(features, (0, 0, 0, pad_len))  # Pad time dim
            windows.append(feat_pad)
            indices.append((0, T))
        else:
            for start in range(0, T - window_size + 1, stride):
                end = start + window_size
                windows.append(features[start:end])
                indices.append((start, end))

            # Handle last window if not covered perfectly
            if T > window_size and (T - window_size) % stride != 0:
                start = T - window_size
                end = T
                windows.append(features[start:end])
                indices.append((start, end))

        if not windows:
            return probs_acc

        # Batch processing of windows
        batch_size = Config.BATCH_SIZE

        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch_wins = windows[i : i + batch_size]
                batch_idxs = indices[i : i + batch_size]

                # Stack: (B, Window, Dim)
                input_tensor = torch.stack(batch_wins).to(self.device)

                # Forward
                _, _, s3_logits = self.model(input_tensor)  # (B, C, Window)

                # Softmax
                s3_probs = F.softmax(s3_logits, dim=1)  # (B, C, Window)

                # Accumulate
                for b in range(len(batch_wins)):
                    start, end = batch_idxs[b]
                    # Transpose to (Window, C)
                    p = s3_probs[b].transpose(0, 1)

                    # If we padded, slice valid part
                    if T < window_size:
                        p = p[:T]

                    probs_acc[start:end] += p
                    count_acc[start:end] += 1.0

        # Average
        final_probs = probs_acc / (count_acc + 1e-8)
        return final_probs

    def generate_submission(self, output_file=None):
        """
        Generates predictions for the test set and saves to CSV.
        Args:
            output_file (str): Path to save the submission CSV. Defaults to Config.SUBMISSION_FILE.
        """
        output_file = output_file if output_file else Config.SUBMISSION_FILE

        # Load Test Dataset
        # load_cached_data=True ensures we use the cache mechanism defined in Dataset
        test_dataset = GestureDataset(
            split="test", mode="inference", load_cached_data=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Generating predictions for {len(test_dataset)} test sequences...")

        results = []

        with torch.no_grad():
            for features, _, sample_id_tuple in test_loader:
                # Unpack batch of size 1
                features = features.squeeze(0).to(self.device)  # (T, D)
                sample_id = sample_id_tuple[0]

                # Inference
                probs = self.predict_sliding_window(features)  # (T, C)

                # Decode
                pred_labels = torch.argmax(probs, dim=1).cpu().numpy()

                # Run Length Encoding to get sequence of gestures
                gesture_sequence = run_length_encoding(pred_labels)

                # Format string: "SessionID,Label1,Label2,..."
                seq_str = ",".join(map(str, gesture_sequence))

                # If sequence is empty, line is just "SessionID" (or "SessionID," depending on strictness, usually CSV handles empty cols)
                # Based on example: "Session00001,2,12,3"
                if seq_str:
                    line = f"{sample_id},{seq_str}"
                else:
                    line = f"{sample_id}"

                results.append(line)

        # Write to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {output_file}")


def run_inference():
    """
    Main entry point for inference.
    """
    engine = InferenceEngine()
    engine.generate_submission()
