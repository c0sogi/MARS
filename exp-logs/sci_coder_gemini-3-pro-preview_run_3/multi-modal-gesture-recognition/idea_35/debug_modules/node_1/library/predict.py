import torch
import os
import logging
import numpy as np
from library.config import config
from library.model import RHCKN
from library.dataset import get_dataloaders
from library.utils import process_predictions_for_submission, setup_logger

# Initialize logger
logger = setup_logger()


class SlidingWindowPredictor:
    """
    Handles inference for the RHC-KN model using a sliding window approach.
    Reconstructs full sequence probabilities from overlapping window predictions,
    decodes them into gesture sequences, and generates the submission file.
    """

    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path (str, optional): Path to the saved model weights. Defaults to config.MODEL_SAVE_PATH.
            device (torch.device, optional): Device to run inference on. Defaults to auto-detect.
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_path = model_path if model_path else config.MODEL_SAVE_PATH

        # Initialize Model
        self.model = RHCKN().to(self.device)
        self.model.eval()

        # Load Weights
        if os.path.exists(self.model_path):
            logger.info(f"Loading model weights from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            logger.warning(
                f"Model file not found at {self.model_path}. Using random initialization (expect poor results)."
            )

    def _reconstruct_sequences(self, loader):
        """
        Reconstructs full sequence probabilities from windowed predictions.
        Accumulates probabilities from overlapping windows and normalizes by overlap count.

        Args:
            loader (DataLoader): The test data loader.

        Returns:
            list of torch.Tensor: A list where each element is a (T, NumClasses) tensor
                                  containing the averaged probabilities for a sample.
        """
        dataset = loader.dataset

        # Initialize buffers for each sample in the dataset
        # We need to access the raw samples to know the full length
        sample_preds = []
        sample_counts = []

        for s in dataset.samples:
            # Determine length from skeleton data (T, J, 3)
            length = s["skeleton"].shape[0]
            # Create buffers on device to avoid frequent transfers
            sample_preds.append(
                torch.zeros(length, config.NUM_CLASSES, device=self.device)
            )
            sample_counts.append(torch.zeros(length, 1, device=self.device))

        # We assume the loader iterates sequentially through the windows as defined in dataset.windows
        # The dataset.windows list maps linear indices to (sample_idx, start, end)
        global_window_idx = 0

        logger.info("Starting inference loop...")

        with torch.no_grad():
            for features, _ in loader:
                features = features.to(self.device)

                # Forward pass - use Stage 3 (Final Refinement) for prediction
                # The model returns logits1, logits2, logits3
                _, _, logits3 = self.model(features)
                probs = torch.softmax(logits3, dim=-1)  # (Batch, Window, Classes)

                batch_size = features.size(0)

                for i in range(batch_size):
                    # Safety check
                    if global_window_idx >= len(dataset.windows):
                        break

                    # Get window metadata: which sample and what range does this window cover?
                    s_idx, start, end = dataset.windows[global_window_idx]

                    # Determine valid length
                    # The window in dataset.windows is (start, end) indices of the original sample.
                    # The model output is always config.WINDOW_SIZE.
                    # If end - start < config.WINDOW_SIZE, the input was padded in dataset.__getitem__.
                    # We only want to accumulate the valid (non-padded) predictions corresponding to the original sequence.
                    valid_len = end - start

                    # Extract valid probabilities from the window output
                    valid_probs = probs[i, :valid_len, :]

                    # Accumulate predictions into the global buffer
                    sample_preds[s_idx][start:end] += valid_probs
                    sample_counts[s_idx][start:end] += 1.0

                    global_window_idx += 1

        # Normalize accumulated probabilities by the overlap count
        final_sequences = []
        for p, c in zip(sample_preds, sample_counts):
            # Avoid division by zero (though count should be >= 1 for all valid frames)
            c = c.clamp(min=1.0)
            avg_probs = p / c
            final_sequences.append(avg_probs)

        return final_sequences

    def generate_predictions(self, test_loader):
        """
        Generates final predictions for the test set.

        Args:
            test_loader (DataLoader): DataLoader for the test set.

        Returns:
            list of str: List of formatted strings "SessionID,Label1,Label2,..."
        """
        reconstructed_probs = self._reconstruct_sequences(test_loader)
        dataset = test_loader.dataset

        results = []

        logger.info("Decoding predictions...")

        for i, probs in enumerate(reconstructed_probs):
            sample_id = dataset.samples[i]["id"]

            # Decode: Argmax -> RLE -> Filter Short -> Remove Background
            # probs is (T, NumClasses)
            frame_preds = torch.argmax(probs, dim=1).cpu().numpy()

            # Use utility function for consistent post-processing
            pred_labels = process_predictions_for_submission(
                frame_preds, background_class=0
            )

            # Format string: "SessionID,Label1,Label2,..."
            if not pred_labels:
                # Handle case with no gestures detected
                pred_str = ""
            else:
                pred_str = ",".join(map(str, pred_labels))

            results.append(f"{sample_id},{pred_str}")

        return results

    def save_submission(self, results, output_path=None):
        """
        Saves the prediction results to a CSV file.

        Args:
            results (list of str): Formatted prediction strings.
            output_path (str, optional): Path to save the file. Defaults to config.SUBMISSION_FILE.
        """
        if output_path is None:
            output_path = config.SUBMISSION_FILE

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.info(f"Saving submission to {output_path}")
        with open(output_path, "w") as f:
            for line in results:
                f.write(line + "\n")


def run_prediction(batch_size=None):
    """
    Main entry point to run the prediction pipeline.
    """
    # Ensure reproducibility
    config.set_seed()

    # Get DataLoaders (only need test_loader here)
    # We pass None for batch_size to use config default if not provided
    _, _, test_loader = get_dataloaders(batch_size=batch_size)

    # Initialize Predictor
    predictor = SlidingWindowPredictor()

    # Generate Predictions
    results = predictor.generate_predictions(test_loader)

    # Save Results
    predictor.save_submission(results)

    logger.info("Prediction pipeline completed successfully.")
