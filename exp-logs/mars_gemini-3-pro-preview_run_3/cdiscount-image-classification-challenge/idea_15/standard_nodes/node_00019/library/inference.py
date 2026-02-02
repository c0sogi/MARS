import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import HierarchyMapper
from library.model import DeepFeatureCascade
from library.feature_dataset import RamFeatureDataset


class EnsemblePredictor:
    """
    Manages the inference process using an ensemble of DeepFeatureCascade models.
    Loads pre-computed features, performs batched inference, aggregates predictions,
    and generates the submission CSV.
    """

    def __init__(self, model_paths, device=None):
        """
        Args:
            model_paths (list): List of file paths to the trained model checkpoints (.pth).
            device (str, optional): Device to run inference on ('cuda' or 'cpu').
        """
        self.model_paths = model_paths
        self.device = device if device else Config.DEVICE
        self.mapper = HierarchyMapper(load_cached_data=True)
        self.models = []
        self._load_models()

    def _load_models(self):
        """
        Initializes and loads all models in the ensemble onto the device.
        """
        print(f"Loading {len(self.model_paths)} models for ensemble inference...")
        for path in self.model_paths:
            if not os.path.exists(path):
                print(f"Warning: Model checkpoint not found at {path}. Skipping.")
                continue

            model = DeepFeatureCascade()
            # Load weights
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)

            model.to(self.device)
            model.eval()
            self.models.append(model)

        if not self.models:
            raise RuntimeError(
                "No models were successfully loaded. Cannot proceed with inference."
            )
        print("All models loaded successfully.")

    def predict(
        self, test_feature_path, test_id_path, batch_size=Config.TRAIN_BATCH_SIZE
    ):
        """
        Runs inference on the test dataset.

        Args:
            test_feature_path (str): Path to the .npy file containing test features.
            test_id_path (str): Path to the .npy file containing test product IDs.
            batch_size (int): Batch size for inference.

        Returns:
            pd.DataFrame: DataFrame containing '_id' and 'category_id'.
        """
        # 1. Load Data
        print(f"Loading test features from {test_feature_path}...")
        dataset = RamFeatureDataset(
            feature_path=test_feature_path, label_path=None, mode="test"
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Crucial to maintain alignment with IDs
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Loading test IDs from {test_id_path}...")
        test_ids = np.load(test_id_path)

        if len(test_ids) != len(dataset):
            raise ValueError(
                f"Mismatch between number of features ({len(dataset)}) and IDs ({len(test_ids)})."
            )

        # 2. Inference Loop
        print(
            f"Starting inference on {len(dataset)} samples using batch size {batch_size}..."
        )

        all_preds = []

        with torch.no_grad():
            for i, features in enumerate(loader):
                features = features.to(self.device)

                # Aggregate probabilities from all models
                avg_probs = None

                for model in self.models:
                    # Forward pass
                    _, _, l3_logits = model(features)

                    # Compute Softmax probabilities for the target level (L3)
                    probs = F.softmax(l3_logits, dim=1)

                    if avg_probs is None:
                        avg_probs = probs
                    else:
                        avg_probs += probs

                # Average across ensemble
                avg_probs /= len(self.models)

                # Get prediction (L3 index)
                batch_preds = torch.argmax(avg_probs, dim=1).cpu().numpy()
                all_preds.append(batch_preds)

                if (i + 1) % 50 == 0:
                    print(f"Processed batch {i + 1}/{len(loader)}")

        # 3. Post-processing
        print("Inference complete. Processing predictions...")
        final_l3_indices = np.concatenate(all_preds, axis=0)

        # Map L3 indices back to original category_ids
        final_category_ids = self.mapper.get_submission_ids(final_l3_indices)

        # Create DataFrame
        submission_df = pd.DataFrame(
            {"_id": test_ids, "category_id": final_category_ids}
        )

        return submission_df

    def generate_submission(
        self, output_path=Config.SUBMISSION_PATH, batch_size=Config.TRAIN_BATCH_SIZE
    ):
        """
        Wrapper method to run prediction and save the result to CSV.

        Args:
            output_path (str): Path to save the submission CSV.
            batch_size (int): Batch size for inference.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Run prediction
        df = self.predict(
            test_feature_path=Config.TEST_FEATURES,
            test_id_path=Config.TEST_IDS,
            batch_size=batch_size,
        )

        # Save
        print(f"Saving submission to {output_path}...")
        df.to_csv(output_path, index=False)
        print("Submission saved successfully.")
