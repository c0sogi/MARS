import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed
from library.model import SD_DGN
from library.data_loader import process_sequence


class InferenceEngine:
    """
    Engine for performing inference using the SD-DGN model.
    Handles model loading, config patching, and sequence prediction.
    """

    def __init__(self, checkpoint_path=None):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        if checkpoint_path is None:
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

        self.checkpoint_path = checkpoint_path
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        """
        Loads the model from checkpoint.
        Dynamically patches Config.NUM_CLASSES if a mismatch is detected in the checkpoint.
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")

        state_dict = torch.load(self.checkpoint_path, map_location=self.device)

        # Robustness: Check if Config.NUM_CLASSES matches checkpoint
        # Stage 1 classifier weight: [num_classes, hidden*2]
        if "stage1.classifier.weight" in state_dict:
            ckpt_classes = state_dict["stage1.classifier.weight"].shape[0]
            if ckpt_classes != Config.NUM_CLASSES:
                print(
                    f"Warning: Config.NUM_CLASSES ({Config.NUM_CLASSES}) does not match checkpoint ({ckpt_classes}). Patching Config."
                )
                Config.NUM_CLASSES = ckpt_classes

        # Instantiate model with potentially patched Config
        model = SD_DGN().to(self.device)
        model.load_state_dict(state_dict)
        return model

    def _decode_predictions(self, frame_preds):
        """
        Decodes frame-wise predictions into a list of gesture IDs.
        Applies Run-Length Encoding and Min Duration Filtering.
        Assumes Class 0 is background and ignores it.
        """
        if len(frame_preds) == 0:
            return []

        # 1. Run-Length Encoding
        runs = []
        current_label = frame_preds[0]
        current_len = 1

        for i in range(1, len(frame_preds)):
            if frame_preds[i] == current_label:
                current_len += 1
            else:
                runs.append((current_label, current_len))
                current_label = frame_preds[i]
                current_len = 1
        runs.append((current_label, current_len))

        # 2. Filter
        gesture_ids = []
        for label, length in runs:
            # Ignore background (Class 0)
            if label == 0:
                continue

            # Filter short segments
            if length >= Config.MIN_GESTURE_LENGTH:
                gesture_ids.append(int(label))

        return gesture_ids

    def predict_sequence(self, features):
        """
        Performs inference on a single sequence features array.
        Args:
            features (np.ndarray): Input features of shape [Time, Channels]
        Returns:
            list: Ordered list of recognized gesture IDs.
        """
        # Prepare input: [Time, Channels] -> [1, Channels, Time]
        x = torch.from_numpy(features).float()
        x = x.transpose(0, 1).unsqueeze(0)  # [1, C, T]
        x = x.to(self.device)

        with torch.no_grad():
            # Forward pass
            # Returns tuple (logits1, logits2, logits3)
            _, _, logits3 = self.model(x)

            # Logits: [1, Classes, Time]
            # Argmax
            preds = torch.argmax(logits3, dim=1)  # [1, Time]
            preds = preds.squeeze(0).cpu().numpy()  # [Time]

        return self._decode_predictions(preds)


def process_test_data_cached(metadata_path, cache_dir, load_cached_data=True):
    """
    Processes test data with caching mechanism.
    Returns a dictionary mapping sample_id -> features (np.ndarray).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "test_features.parquet.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached test features from {cache_file}...")
        try:
            loaded = np.load(cache_file, allow_pickle=True)
            # Reconstruct dictionary from npz
            return {k: loaded[k] for k in loaded.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print(f"Processing test data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    data_map = {}

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        # process_sequence returns (features, labels)
        # We pass is_train=False to avoid rotation augmentation
        feat, _ = process_sequence(row, is_train=False)

        if feat is not None:
            data_map[sample_id] = feat
        else:
            # Handle empty/error case (e.g., video too short)
            # We skip adding it to the map; generation loop handles missing keys
            pass

    # Save to cache
    print(f"Saving test features to {cache_file}...")
    np.savez_compressed(cache_file, **data_map)

    return data_map


def generate_submission(load_cached_data=True):
    """
    Generates the submission file for the test set.
    Loads the best model, processes test data, and writes predictions to CSV.
    """
    # Initialize Engine
    try:
        engine = InferenceEngine()
    except Exception as e:
        print(f"Failed to initialize InferenceEngine: {e}")
        return

    # Load Data
    # We use the working directory for caching
    cache_dir = Config.WORK_DIR
    test_data = process_test_data_cached(
        Config.TEST_METADATA_PATH, cache_dir, load_cached_data
    )

    # Load Metadata to ensure order and completeness
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    results = []

    print("Running inference on test set...")
    for _, row in df_test.iterrows():
        sample_id = row["sample_id"]

        if sample_id in test_data:
            features = test_data[sample_id]
            predicted_ids = engine.predict_sequence(features)
        else:
            # Fallback for failed/missing samples
            predicted_ids = []

        # Format: SessionID,id1,id2,...
        # Note: Trailing comma is allowed/expected if list is empty?
        # The example shows "Session00001,2,12,3".
        if predicted_ids:
            pred_str = ",".join(map(str, predicted_ids))
            line = f"{sample_id},{pred_str}"
        else:
            # If empty, just the session ID
            line = f"{sample_id},"

        results.append(line)

    # Write submission
    out_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Submission saved to {out_path}")
