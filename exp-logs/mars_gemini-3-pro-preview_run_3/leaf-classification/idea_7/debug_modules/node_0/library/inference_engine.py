import os
import pandas as pd
import numpy as np
from library.config import Config
from library.cv_runner import StratifiedEnsembleTrainer
from library.data_loader import LeafDataManager
from library.utils import clip_probabilities


class EnsemblePredictor:
    """
    Manages the inference phase of the pipeline.
    Responsible for generating predictions using the trained ensemble,
    formatting the output according to competition requirements, and saving the submission file.
    """

    def __init__(self):
        """
        Initialize the predictor with necessary components.
        """
        self.trainer = StratifiedEnsembleTrainer()
        self.data_manager = LeafDataManager()

    def create_submission(
        self, dino_features, conv_features, tabular_features, test_ids
    ):
        """
        Generates predictions for the test set using the ensemble of trained models,
        formats the results, and saves them to a CSV file.

        Args:
            dino_features (np.ndarray): Test global geometry features (N, D1).
            conv_features (np.ndarray): Test local margin features (N, D2).
            tabular_features (np.ndarray): Test tabular features (N, 192).
            test_ids (np.ndarray): Array of test image IDs (N,).

        Returns:
            None
        """
        print("Starting inference process...")

        # 1. Generate Predictions
        # Use the trainer's predict_test method which handles:
        # - Loading all K saved pipeline pickles
        # - Predicting with each model
        # - Averaging the probabilities (Soft Voting)
        avg_probs = self.trainer.predict_test(
            dino_features, conv_features, tabular_features
        )

        # 2. Apply Probability Clipping
        # As per task requirements: max(min(p, 1-10^-15), 10^-15)
        # This prevents infinite penalties in Log Loss calculation
        print(
            f"Clipping probabilities to range [{Config.PROB_CLIP_MIN}, {Config.PROB_CLIP_MAX}]..."
        )
        clipped_probs = clip_probabilities(
            avg_probs, clip_min=Config.PROB_CLIP_MIN, clip_max=Config.PROB_CLIP_MAX
        )

        # 3. Retrieve Class Names
        # Get the list of class names in the order used by the LabelEncoder
        classes = self.data_manager.get_classes()

        # Validation check
        if len(classes) != clipped_probs.shape[1]:
            raise ValueError(
                f"Mismatch between number of classes ({len(classes)}) and "
                f"prediction output dimensions ({clipped_probs.shape[1]})."
            )

        # 4. Construct Submission DataFrame
        print("Formatting submission DataFrame...")
        # Initialize with IDs
        submission_data = {"id": test_ids}

        # Add a column for each species with its corresponding probability
        for i, class_name in enumerate(classes):
            submission_data[class_name] = clipped_probs[:, i]

        submission_df = pd.DataFrame(submission_data)

        # 5. Save to Disk
        # Ensure the directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

        print(f"Submission saved successfully to: {Config.SUBMISSION_PATH}")
        print(f"Final Submission Shape: {submission_df.shape}")
