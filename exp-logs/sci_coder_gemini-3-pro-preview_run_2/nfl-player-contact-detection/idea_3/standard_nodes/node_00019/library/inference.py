import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import compute_mcc


class InferenceEngine:
    """
    Manages post-training inference, threshold optimization, and submission generation.
    """

    def __init__(self, model, device=Config.DEVICE):
        """
        Args:
            model (torch.nn.Module): The trained model architecture.
            device (str): Device to run inference on ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.device = device

    def load_weights(self, path=None):
        """
        Loads model weights from the specified path or the default cache location.

        Args:
            path (str, optional): Path to the .pth file. Defaults to Config.CACHE_DIR/best_model.pth.
        """
        if path is None:
            path = os.path.join(Config.CACHE_DIR, "best_model.pth")

        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model weights from {path}")
        else:
            print(
                f"Warning: Weights file not found at {path}. Using current model state."
            )

    def optimize_threshold(self, val_loader, step=0.01):
        """
        Performs a grid search on the validation set to find the decision threshold
        that maximizes the MCC.

        Args:
            val_loader (DataLoader): DataLoader for the validation set.
            step (float): Step size for the threshold grid search.

        Returns:
            float: The optimal threshold value.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        # Collect predictions and targets
        with torch.no_grad():
            for batch in val_loader:
                # Handle case where loader returns (X, y)
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)

                    all_preds.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        # Grid Search
        best_mcc = -1.0
        best_threshold = 0.5

        # Search range from roughly 0.0 to 1.0
        thresholds = np.arange(step, 1.0, step)

        for t in thresholds:
            # Apply threshold
            bin_preds = (all_preds >= t).astype(int)

            # Compute MCC
            mcc = compute_mcc(all_targets, bin_preds)

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = t

        print(f"Optimization Complete. Best MCC: {best_mcc}")
        print(f"Optimal Threshold: {best_threshold}")

        return best_threshold

    def predict_test(self, test_loader, threshold, output_path=Config.SUBMISSION_PATH):
        """
        Generates predictions for the test set using the specified threshold and saves
        to a CSV file.

        Args:
            test_loader (DataLoader): DataLoader for the test set.
            threshold (float): Decision threshold for binary classification.
            output_path (str): Path to save the submission CSV.
        """
        self.model.eval()
        all_preds = []

        # Collect raw probabilities
        with torch.no_grad():
            for batch in test_loader:
                # Test loader might return just X or (X, y) depending on implementation
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                all_preds.append(outputs.cpu().numpy())

        all_preds = np.concatenate(all_preds).flatten()

        # Apply Threshold
        binary_preds = (all_preds >= threshold).astype(int)

        # Retrieve contact_ids
        # Assumes the dataset underlying the loader has a 'contact_ids' attribute
        if hasattr(test_loader.dataset, "contact_ids"):
            contact_ids = test_loader.dataset.contact_ids
        else:
            raise AttributeError(
                "Test dataset must have 'contact_ids' attribute to generate submission."
            )

        # Validation
        if len(contact_ids) != len(binary_preds):
            raise ValueError(
                f"Size mismatch: {len(contact_ids)} IDs vs {len(binary_preds)} predictions."
            )

        # Create DataFrame
        submission_df = pd.DataFrame(
            {"contact_id": contact_ids, "contact": binary_preds}
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
