import os
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.model import SRDGN
from library.data_loader import GestureDataset
from library.utils import set_seed, setup_logger, run_length_encoding

# Setup logger for the prediction module
logger = setup_logger("predict_module", os.path.join(Config.WORKING_DIR, "predict.log"))


class PostProcessor:
    """
    Handles the conversion of raw model predictions into the submission format.
    """

    @staticmethod
    def process(predictions, min_length=5):
        """
        Applies Run-Length Encoding to frame-wise predictions and formats them as a CSV string.

        Args:
            predictions (np.ndarray): Array of frame-wise class labels.
            min_length (int): Minimum duration for a gesture to be considered valid.

        Returns:
            str: Comma-separated string of gesture IDs.
        """
        # Use the utility function for RLE
        segments = run_length_encoding(predictions, min_length=min_length)
        return ",".join(map(str, segments))


class InferenceEngine:
    """
    Manages the inference process using the SR-DGN model.
    """

    def __init__(self, checkpoint_path, device=None):
        """
        Initialize the inference engine.

        Args:
            checkpoint_path (str): Path to the trained model weights.
            device (torch.device, optional): Compute device. Defaults to auto-detect.
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize model architecture: 180 (kinematics) + 13 (audio) = 193 input dim
        self.model = SRDGN(input_dim=193, num_classes=Config.NUM_CLASSES).to(
            self.device
        )

        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning(
                f"Checkpoint not found at {checkpoint_path}. Using random weights (expect poor performance)."
            )

        self.model.eval()

    def run(self, data_loader):
        """
        Runs inference on the provided data loader.

        Args:
            data_loader (DataLoader): DataLoader for the test set.

        Returns:
            list: A list of formatted strings "SessionID,Label1,Label2,..."
        """
        results = []
        logger.info(f"Starting inference on {len(data_loader)} samples...")

        with torch.no_grad():
            for batch in data_loader:
                features = batch["features"].to(self.device)
                # Batch size is 1 for test inference to preserve full sequence context
                sample_id = batch["id"][0]

                # Forward pass - Stage 3 provides the final refined predictions
                _, _, logits3 = self.model(features)

                # Get frame-wise predictions
                preds = torch.argmax(logits3, dim=2).squeeze(0).cpu().numpy()

                # Post-process predictions
                pred_str = PostProcessor.process(
                    preds, min_length=Config.MIN_GESTURE_LENGTH
                )

                results.append(f"{sample_id},{pred_str}")

        return results


def predict(load_cached_data=True):
    """
    Main entry point for generating predictions.

    Args:
        load_cached_data (bool): Whether to use cached dataset files if available.
    """
    set_seed(Config.SEED)

    # Define paths
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    test_csv = os.path.join(Config.METADATA_DIR, "test.csv")
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Initialize Test Dataset
    # We rely on the Dataset class to handle caching logic via the load_cached_data flag
    test_dataset = GestureDataset(
        test_csv, mode="test", load_cached_data=load_cached_data
    )

    # Initialize DataLoader
    # Batch size must be 1 to handle variable length sequences without padding artifacts during inference
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    # Run Inference
    engine = InferenceEngine(checkpoint_path)
    results = engine.run(test_loader)

    # Save Results
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    with open(submission_path, "w") as f:
        for line in results:
            f.write(line + "\n")

    logger.info(f"Submission saved to {submission_path}")
