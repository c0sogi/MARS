import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
from scipy.interpolate import interp1d
from tqdm import tqdm

from library.config import Config
from library.utils import load_mat_safe, rle_encode, get_device
from library.model import RLK_RN


class InferenceEngine:
    """
    Handles inference for the Residual Log-Kinematic Refinement Network.
    Includes feature extraction, sliding window prediction, and submission generation.
    """

    def __init__(self, model_path=None):
        self.device = get_device()
        self.model = RLK_RN().to(self.device)

        # Load weights if provided, otherwise expect them to be loaded later
        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)
        elif os.path.exists(Config.BEST_MODEL_PATH):
            self.load_weights(Config.BEST_MODEL_PATH)
        else:
            print("Warning: No model weights found. Inference will use random weights.")

        self.model.eval()

    def load_weights(self, path):
        """Loads model state dictionary."""
        print(f"Loading model weights from {path}...")
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def _process_audio_single(self, audio_path, target_frames):
        """
        Extracts MFCCs for a single audio file and aligns to target frames.
        """
        try:
            if not os.path.exists(audio_path):
                return np.zeros((target_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

            y, sr = librosa.load(audio_path, sr=Config.AUDIO_SAMPLE_RATE)
            if len(y) == 0:
                return np.zeros((target_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=Config.AUDIO_N_MFCC)
            n_mfcc, n_audio_frames = mfcc.shape

            x_old = np.linspace(0, 1, n_audio_frames)
            x_new = np.linspace(0, 1, target_frames)

            f = interp1d(x_old, mfcc, axis=1, kind="linear", fill_value="extrapolate")
            mfcc_resampled = f(x_new).T
            return mfcc_resampled.astype(np.float32)
        except Exception:
            return np.zeros((target_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

    def _extract_features_from_row(self, row):
        """
        Extracts and fuses features for a single metadata row.
        Returns: (T, InputDim) numpy array or None if invalid.
        """
        # 1. Load Skeleton
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        skel = load_mat_safe(mat_path)  # (T, 20, 3)

        if skel is None:
            return None

        num_frames = skel.shape[0]
        if num_frames < 5:  # Minimal length check
            return None

        # 2. Normalize (mm -> m)
        skel = skel / 1000.0

        # 3. Compute Derivatives
        # Pad start to compute derivatives for first frames
        # We need T frames of output.
        # Velocity needs T+1 input, Acceleration needs T+2 input.
        # We replicate the first frame twice.
        padding = np.repeat(skel[0:1], 2, axis=0)
        padded_skel = np.concatenate([padding, skel], axis=0)  # (T+2, 20, 3)

        velocity = np.diff(padded_skel, axis=0)  # (T+1, 20, 3)
        acceleration = np.diff(velocity, axis=0)  # (T, 20, 3)

        # Align features
        # Pos: Use original T frames (padded_skel[2:])
        pos_feat = padded_skel[2:].reshape(num_frames, -1)
        # Vel: Use last T frames of velocity (velocity[1:])
        vel_feat = velocity[1:].reshape(num_frames, -1)
        # Acc: Use acceleration (acceleration is already T)
        acc_feat = acceleration.reshape(num_frames, -1)

        # 4. Audio
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])
        audio_feat = self._process_audio_single(audio_path, num_frames)

        # 5. Fusion
        features = np.concatenate([pos_feat, vel_feat, acc_feat, audio_feat], axis=1)
        return features.astype(np.float32)

    def get_test_features(self, load_cached_data=True):
        """
        Loads or computes features for the entire test set.
        Returns a dictionary {sample_id: features_array}.
        """
        cache_path = os.path.join(Config.CACHE_DIR, "test_features_dict.npz")

        # Try loading cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached test features from {cache_path}...")
            try:
                loaded = np.load(cache_path, allow_pickle=True)
                # Convert back to dict
                feature_dict = {k: v for k, v in loaded.items()}
                return feature_dict
            except Exception as e:
                print(f"Cache load failed: {e}. Reprocessing...")

        # Process from scratch
        print("Processing test features...")
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        feature_dict = {}

        for _, row in tqdm(
            df_test.iterrows(), total=len(df_test), desc="Extracting Features"
        ):
            sample_id = row["sample_id"]
            feats = self._extract_features_from_row(row)
            if feats is not None:
                feature_dict[sample_id] = feats
            else:
                # Handle edge case of missing/corrupt data by creating dummy features
                # This ensures we can still output a prediction line
                print(
                    f"Warning: Could not extract features for {sample_id}. Using zeros."
                )
                dummy_len = 100
                feature_dict[sample_id] = np.zeros(
                    (dummy_len, Config.INPUT_DIM), dtype=np.float32
                )

        # Save cache
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        np.savez(cache_path, **feature_dict)
        print(f"Saved test features to {cache_path}")

        return feature_dict

    def predict_sequence(self, features):
        """
        Performs sliding window inference on a single sequence.
        Args:
            features: (T, InputDim) numpy array.
        Returns:
            probs: (T, NumClasses) numpy array of probabilities.
        """
        T = features.shape[0]
        window_size = Config.WINDOW_SIZE
        stride = Config.STRIDE  # 32 for 50% overlap

        # Handle short sequences
        if T < window_size:
            pad_len = window_size - T
            padding = np.zeros((pad_len, features.shape[1]), dtype=features.dtype)
            features = np.concatenate([features, padding], axis=0)
            T = features.shape[0]

        # Prepare windows
        windows = []
        indices = []

        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            window = features[start:end]
            windows.append(window)
            indices.append((start, end))

        # Handle last window if not covered perfectly
        if indices[-1][1] < T:
            start = T - window_size
            end = T
            window = features[start:end]
            windows.append(window)
            indices.append((start, end))

        # Batch inference
        batch_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(
            self.device
        )

        probs_accum = np.zeros((T, Config.NUM_CLASSES), dtype=np.float32)
        counts_accum = np.zeros((T, 1), dtype=np.float32)

        with torch.no_grad():
            # Process in chunks to avoid OOM on very long sequences
            chunk_size = 64
            for i in range(0, len(batch_tensor), chunk_size):
                batch = batch_tensor[i : i + chunk_size]

                # Forward pass
                outputs = self.model(batch)
                # Use Stage 3 output (index 2)
                log_probs = outputs[2]
                probs = torch.exp(log_probs).cpu().numpy()  # (B, Window, Classes)

                # Accumulate
                for j, prob_map in enumerate(probs):
                    global_idx = i + j
                    start, end = indices[global_idx]
                    probs_accum[start:end] += prob_map
                    counts_accum[start:end] += 1

        # Average
        # Avoid division by zero
        counts_accum[counts_accum == 0] = 1.0
        final_probs = probs_accum / counts_accum

        return final_probs

    def generate_submission(
        self, output_path=Config.SUBMISSION_PATH, load_cached_data=True
    ):
        """
        Generates the submission CSV file.
        """
        print("Generating submission...")

        # Get features
        feature_dict = self.get_test_features(load_cached_data=load_cached_data)

        results = []

        # Sort keys to ensure deterministic order (though CSV handles ID mapping)
        sample_ids = sorted(feature_dict.keys())

        for sample_id in tqdm(sample_ids, desc="Predicting"):
            features = feature_dict[sample_id]

            # Predict
            probs = self.predict_sequence(features)

            # Decode
            frame_preds = np.argmax(probs, axis=1)

            # RLE and Filter Background
            gesture_ids = rle_encode(
                frame_preds, background_label=Config.BACKGROUND_LABEL
            )

            # Format string
            # "SessionID,Label1,Label2,..."
            label_str = ",".join(map(str, gesture_ids))
            line = f"{sample_id},{label_str}"
            results.append(line)

        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {output_path}")
