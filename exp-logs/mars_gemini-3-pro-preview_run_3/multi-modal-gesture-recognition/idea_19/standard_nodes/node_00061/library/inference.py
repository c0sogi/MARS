import os
import torch
import numpy as np
import torch.nn.functional as F
from library.config import Config, seed_everything
from library.model import KC_IRN
from library.data_utils import (
    load_dataset_and_cache,
    get_feature_vector,
)


class InferenceEngine:
    """
    Engine for performing inference using the KC-IRN model.
    Handles model loading, sliding window prediction, temporal ensembling,
    and submission generation.
    """

    def __init__(self):
        """
        Initialize the Inference Engine.
        Sets random seeds, loads the model architecture, and restores weights.
        """
        seed_everything(Config.SEED)
        self.device = Config.DEVICE

        # Initialize Model
        self.model = KC_IRN().to(self.device)
        self.load_model()

    def load_model(self):
        """
        Loads the best model weights from the cache directory.
        """
        model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. Please train the model first."
            )

        print(f"Loading model from {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_sequence(self, features):
        """
        Performs Sliding Window Inference on a single sequence.

        Args:
            features (np.ndarray): Input features of shape (T, Input_Dim).

        Returns:
            np.ndarray: Frame-wise label predictions of shape (T,).
        """
        T = features.shape[0]
        window_size = Config.WINDOW_SIZE
        stride = Config.STRIDE_TEST  # 32 for 50% overlap
        num_classes = Config.NUM_CLASSES

        # Accumulators for Temporal Ensembling
        probs_sum = np.zeros((T, num_classes), dtype=np.float32)
        counts = np.zeros((T, 1), dtype=np.float32)

        # Handle sequences shorter than window size
        if T < window_size:
            pad_len = window_size - T
            # Pad end with zeros
            padded_feat = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")

            # Convert to tensor
            tensor_feat = (
                torch.from_numpy(padded_feat).float().unsqueeze(0).to(self.device)
            )  # (1, 64, 193)

            with torch.no_grad():
                outputs = self.model(tensor_feat)
                # Use Stage 3 output for final prediction
                logits = outputs["logits_s3"]
                probs = F.softmax(logits, dim=2).cpu().numpy()[0]  # (64, 21)

            # Accumulate valid part only
            probs_sum += probs[:T]
            counts += 1.0

        else:
            # Generate sliding window start indices
            starts = list(range(0, T - window_size + 1, stride))

            # Ensure the last frames are covered
            if (T - window_size) % stride != 0:
                starts.append(T - window_size)

            # Process windows
            for start in starts:
                end = start + window_size
                window_feat = features[start:end]

                tensor_feat = (
                    torch.from_numpy(window_feat).float().unsqueeze(0).to(self.device)
                )

                with torch.no_grad():
                    outputs = self.model(tensor_feat)
                    logits = outputs["logits_s3"]
                    probs = F.softmax(logits, dim=2).cpu().numpy()[0]  # (64, 21)

                probs_sum[start:end] += probs
                counts[start:end] += 1.0

        # Compute Average Probabilities
        # Avoid division by zero (counts should be >= 1 everywhere)
        counts[counts == 0] = 1.0
        avg_probs = probs_sum / counts  # (T, 21)

        # Determine labels via Argmax
        pred_labels = np.argmax(avg_probs, axis=1)  # (T,)

        return pred_labels

    def decode_predictions(self, pred_labels):
        """
        Decodes frame-wise labels into a list of gesture IDs.
        Applies Run-Length Encoding and filters out the background class.

        Args:
            pred_labels (np.ndarray): Array of frame labels.

        Returns:
            list: Ordered list of recognized gesture IDs.
        """
        decoded = []
        last_label = None

        for label in pred_labels:
            label = int(label)

            # Skip Background
            if label == Config.BACKGROUND_CLASS_ID:
                last_label = None
                continue

            # Add if different from previous (Run-Length Encoding)
            # If we just transitioned from background (last_label is None), we add.
            # If we transitioned from a different class, we add.
            if label != last_label:
                decoded.append(label)
                last_label = label

        return decoded

    def run_inference(self, load_cached_data=True):
        """
        Main execution method. Loads test data, runs predictions, and saves submission.

        Args:
            load_cached_data (bool): Whether to use cached dataset files.
        """
        print("Initializing Inference Pipeline...")

        # 1. Load Test Data
        # Uses library function to handle polymorphic parsing and caching
        print(f"Loading test data from {Config.TEST_CSV}...")
        test_data = load_dataset_and_cache(
            Config.TEST_CSV, "test", load_cached=load_cached_data
        )

        results = []
        total_samples = len(test_data)
        print(f"Starting inference on {total_samples} sequences...")

        # 2. Iterate and Predict
        for i, item in enumerate(test_data):
            sample_id = item["sample_id"]
            skel_raw = item["skeleton"]
            audio_raw = item["audio"]

            # Feature Engineering: Kinematics + Fusion (No Augmentation)
            features = get_feature_vector(skel_raw, audio_raw, augment=False)

            # Predict
            pred_labels = self.predict_sequence(features)

            # Decode
            gesture_list = self.decode_predictions(pred_labels)

            # Format Result
            label_str = ",".join(map(str, gesture_list))
            results.append(f"{sample_id},{label_str}")

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{total_samples} samples...")

        # 3. Save Submission
        out_path = Config.SUBMISSION_FILE
        # Ensure directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Inference complete. Submission saved to {out_path}")
