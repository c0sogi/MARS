import os
import torch
import pandas as pd
from library.config import Config, set_seed
from library.model import BiLSTMClassifier
from library.utils import decode_predictions
from library.data_loader import get_dataloaders


class Predictor:
    """
    Handles inference on the test dataset using a trained Bi-LSTM model.
    """

    def __init__(self, model_class, checkpoint_path, device):
        """
        Initialize the predictor.

        Args:
            model_class (nn.Module class): The class of the model architecture.
            checkpoint_path (str): Path to the saved model weights.
            device (torch.device): Device to run inference on.
        """
        self.device = device
        self.model = model_class().to(self.device)

        if os.path.exists(checkpoint_path):
            print(f"Loading model weights from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Checkpoint {checkpoint_path} not found. Using random weights."
            )

        self.model.eval()

    def predict(self, test_loader):
        """
        Run inference on the test loader.

        Args:
            test_loader (DataLoader): DataLoader for the test set.

        Returns:
            list of tuples: [(sample_id, prediction_string), ...]
        """
        results = []

        print("Starting inference...")
        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                sample_ids = batch["sample_ids"]

                # Forward pass
                logits = self.model(features, lengths)

                # Decode predictions
                # Returns list of lists of integers
                decoded_seqs = decode_predictions(logits)

                # Format for submission
                for sample_id, seq in zip(sample_ids, decoded_seqs):
                    # Convert list of ints to comma-separated string
                    pred_str = ",".join(map(str, seq))
                    results.append((sample_id, pred_str))

        return results

    def save_submission(self, predictions, output_path):
        """
        Save predictions to a CSV file in the required format.

        Format: SessionID,label1,label2,...

        Args:
            predictions (list of tuples): List of (sample_id, pred_str).
            output_path (str): Path to save the CSV.
        """
        print(f"Saving submission to {output_path}...")

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            for sample_id, pred_str in predictions:
                if pred_str:
                    line = f"{sample_id},{pred_str}\n"
                else:
                    line = f"{sample_id},\n"  # Handle empty predictions
                f.write(line)

        print("Submission saved successfully.")


def run_inference(
    checkpoint_path=Config.MODEL_CHECKPOINT,
    output_path=Config.SUBMISSION_FILE,
    debug=False,
):
    """
    Main function to run the inference pipeline.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        output_path (str): Path to save the submission file.
        debug (bool): If True, runs on a small subset of data.
    """
    # Set reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Get DataLoaders (we only need test_loader)
    _, _, test_loader = get_dataloaders(debug=debug)

    # Initialize Predictor
    predictor = Predictor(BiLSTMClassifier, checkpoint_path, device)

    # Run Prediction
    predictions = predictor.predict(test_loader)

    # Save Results
    predictor.save_submission(predictions, output_path)
