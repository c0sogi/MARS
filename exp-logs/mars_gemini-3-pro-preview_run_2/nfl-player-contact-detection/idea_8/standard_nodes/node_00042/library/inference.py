import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library import config, utils, model, dataset, data_processing


class InferencePipeline:
    """
    Manages the inference process for the K-CAN model.
    Loads test data, restores the trained model and threshold,
    generates predictions, and saves the submission file.
    """

    def __init__(self):
        self.device = config.get_device()
        self.model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
        self.threshold_path = os.path.join(config.WORKING_DIR, "best_threshold.npy")
        self.submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    def load_data(self, debug=False, load_cached=True):
        """
        Loads and processes the test data using the FeatureEngineer.
        Leverages the FeatureEngineer's caching mechanism and scaler application logic.
        """
        print("Initializing Feature Engineer...")
        engineer = data_processing.FeatureEngineer()

        print("Processing Test Data...")
        # process_dataset returns X_vals, y, ids.
        # For the test set, 'y' contains placeholder zeros (from metadata), which we discard.
        X_test, _, ids = engineer.process_dataset(
            split="test", load_cached_data=load_cached, debug=debug
        )

        return X_test, ids

    def load_model(self):
        """
        Loads the trained K-CAN model architecture and weights.
        """
        print(f"Loading model from {self.model_path}...")
        net = model.KCAN().to(self.device)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        state_dict = torch.load(self.model_path, map_location=self.device)
        net.load_state_dict(state_dict)
        net.eval()
        return net

    def load_threshold(self):
        """
        Loads the optimized decision threshold determined during training.
        Defaults to 0.5 if the file is missing.
        """
        if not os.path.exists(self.threshold_path):
            print(
                f"Threshold file not found at {self.threshold_path}. Defaulting to 0.5."
            )
            return 0.5

        # Load scalar numpy array
        threshold = np.load(self.threshold_path)[0]
        print(f"Loaded optimized threshold: {threshold}")
        return threshold

    def predict(self, net, X_test):
        """
        Runs inference on the test set in batches.
        """
        # Create Dataset with y=None so __getitem__ returns only the inputs
        test_dataset = dataset.ContactSequenceDataset(X_test, y=None)

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        all_probs = []

        print(f"Starting inference on {len(test_dataset)} samples...")
        with torch.no_grad():
            for inputs in test_loader:
                # Unpack tuple inputs (sequence, center_features)
                sequence, center_features = inputs

                # Move to device
                sequence = sequence.to(self.device)
                center_features = center_features.to(self.device)

                # Forward pass
                logits = net((sequence, center_features))
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu().numpy())

        # Concatenate all batch results
        return np.concatenate(all_probs).flatten()

    def generate_submission(self, ids, probs, threshold):
        """
        Applies the threshold to probabilities and saves the submission CSV.
        """
        print("Generating submission file...")

        # Apply threshold to generate binary predictions
        predictions = (probs >= threshold).astype(int)

        # Create DataFrame matching the required format
        submission_df = pd.DataFrame({"contact_id": ids, "contact": predictions})

        # Save to disk
        submission_df.to_csv(self.submission_path, index=False)
        print(f"Submission saved to {self.submission_path}")
        print(f"Total predictions: {len(submission_df)}")
        print(f"Positive predictions: {predictions.sum()}")

    def run(self, debug=False, load_cached=True):
        """
        Executes the full inference pipeline.
        """
        utils.set_seed()

        # 1. Load Data
        X_test, ids = self.load_data(debug=debug, load_cached=load_cached)

        # 2. Load Resources
        net = self.load_model()
        threshold = self.load_threshold()

        # 3. Predict
        probs = self.predict(net, X_test)

        # 4. Save Submission
        self.generate_submission(ids, probs, threshold)
