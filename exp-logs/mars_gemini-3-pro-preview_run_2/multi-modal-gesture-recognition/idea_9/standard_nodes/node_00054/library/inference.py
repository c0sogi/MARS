import os
import torch
import numpy as np
import scipy.ndimage
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_device, setup_logger
from library.model import MDCRCN
from library.data_loader import GestureDataset, collate_fn


def median_filter_smoothing(predictions, kernel_size=Config.MEDIAN_FILTER_KERNEL_SIZE):
    """
    Applies a median filter to the discrete label sequence to smooth out noise.
    Uses Nearest-Neighbor padding to preserve boundary predictions.

    Args:
        predictions (np.ndarray): 1D array of predicted class indices.
        kernel_size (int): Size of the median filter window. Must be odd.

    Returns:
        np.ndarray: Smoothed prediction array.
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # scipy.ndimage.median_filter with mode='nearest' implements nearest-neighbor padding
    smoothed = scipy.ndimage.median_filter(
        predictions, size=kernel_size, mode="nearest"
    )
    return smoothed


def decode_predictions(logits, mask):
    """
    Decodes frame-wise logits into a sequence of gesture IDs.

    Steps:
    1. Extract valid frames based on mask.
    2. Convert logits to class indices (Argmax).
    3. Apply Median Filter smoothing.
    4. Collapse consecutive repetitions.
    5. Remove background class (Label 0).

    Args:
        logits (torch.Tensor): Logits of shape (Frames, Classes).
        mask (torch.Tensor): Binary mask of shape (Frames,).

    Returns:
        List[int]: Ordered list of recognized gesture IDs.
    """
    # 1. Get valid frames based on mask
    valid_len = int(mask.sum().item())
    valid_logits = logits[:valid_len]  # (T, C)

    # 2. Argmax to get labels
    # We use Softmax first conceptually, but Argmax on logits is equivalent
    preds = torch.argmax(valid_logits, dim=1).cpu().numpy()

    # 3. Median Filtering (Smoothing)
    if len(preds) > 0:
        preds = median_filter_smoothing(preds)

    # 4. Collapse repetitions and remove background
    sequence = []
    prev = -1

    for p in preds:
        p = int(p)
        if p != prev:
            # 5. Remove background (Config.BACKGROUND_LABEL is 0)
            if p != Config.BACKGROUND_LABEL:
                sequence.append(p)
            prev = p

    return sequence


class Predictor:
    """
    Handles the inference process for the MD-CRCN model.
    """

    def __init__(self, model_path=Config.BEST_MODEL_PATH):
        self.device = get_device()
        self.logger = setup_logger("MD-CRCN-Inference")
        self.model_path = model_path

        self.logger.info(f"Initializing Predictor on {self.device}")
        self.model = self._load_model()

    def _load_model(self):
        """Loads the MDCRCN model architecture and weights."""
        model = MDCRCN().to(self.device)

        if os.path.exists(self.model_path):
            self.logger.info(f"Loading weights from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            self.logger.warning(
                f"Checkpoint not found at {self.model_path}. Using random weights."
            )

        model.eval()
        return model

    def run_inference(self, load_cached_data=True, limit=None):
        """
        Runs inference on the test set and generates the submission file.

        Args:
            load_cached_data (bool): Whether to use cached preprocessed data.
            limit (int, optional): Limit number of samples for debugging.
        """
        self.logger.info("Loading Test Data...")
        test_dataset = GestureDataset(
            split="test", load_cached_data=load_cached_data, limit=limit
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        results = []
        self.logger.info(f"Starting inference on {len(test_dataset)} samples...")

        with torch.no_grad():
            for batch_idx, (features, _, mask, sample_ids) in enumerate(test_loader):
                features = features.to(self.device)
                mask = mask.to(self.device)

                # Forward Pass
                outputs = self.model(features, mask)

                # Use Stage 3 output for final prediction
                logits = outputs["stage3"]

                # Process batch
                for i in range(len(sample_ids)):
                    # Decode sequence
                    pred_seq = decode_predictions(logits[i], mask[i])

                    # Format: SessionID,label1 label2 ...
                    # Cite debug_lesson_10: Serialize sequence into a single space-separated string
                    # to ensure the CSV has a fixed number of columns (2).
                    seq_str = " ".join(map(str, pred_seq))
                    results.append(f"{sample_ids[i]},{seq_str}")

        self._save_submission(results)

    def _save_submission(self, results):
        """Saves the prediction results to the submission file."""
        output_path = Config.SUBMISSION_PATH

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            # Write Header matching randomPredictions.csv format
            f.write("Id,Sequence\n")
            for line in results:
                f.write(line + "\n")

        self.logger.info(f"Submission saved to {output_path}")
        self.logger.info(f"Total predictions generated: {len(results)}")
