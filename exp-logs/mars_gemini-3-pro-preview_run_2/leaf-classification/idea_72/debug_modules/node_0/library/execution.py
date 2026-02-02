import os
import numpy as np
import pandas as pd
from library.config import SUBMISSION_PATH, RANDOM_SEED
from library.dataset import LeafDataManager
from library.modeling import SMFRIE_Trainer


class ExperimentRunner:
    """
    Orchestrates the Stratified-Manifold Full-Rank Interaction Ensemble (SM-FRIE) strategy.
    Manages data loading, expert selection (Phase 1), and final inference (Phase 2).
    """

    def __init__(self, load_cached_data=True):
        """
        Args:
            load_cached_data (bool): Whether to load pre-processed data from cache if available.
        """
        self.load_cached_data = load_cached_data
        self.data_manager = LeafDataManager()
        self.trainer = SMFRIE_Trainer()

        # Data containers
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def _subsample_data(self, data, n_samples):
        """
        Helper to subsample data dictionaries for debugging or quick runs.

        Args:
            data (dict): Dictionary containing feature arrays (keys: 'global', 'margin', etc.).
            n_samples (int): Number of samples to select.

        Returns:
            dict: Subsampled data dictionary.
        """
        if n_samples is None or n_samples <= 0:
            return data

        # Determine current size based on 'ids'
        if "ids" not in data:
            return data

        current_size = len(data["ids"])
        if current_size <= n_samples:
            return data

        # Set seed for reproducible subsampling
        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(current_size, n_samples, replace=False)

        subset = {}
        for key, value in data.items():
            # Slice numpy arrays
            if isinstance(value, np.ndarray):
                subset[key] = value[indices]
            else:
                subset[key] = value

        return subset

    def load_data(self):
        """
        Loads training, validation, and test data using the LeafDataManager.
        """
        print("Loading data via LeafDataManager...")
        self.data_train, self.data_val, self.data_test = self.data_manager.load_data(
            load_cached_data=self.load_cached_data
        )

    def run_selection_phase(self, max_train_samples=None):
        """
        Phase 1: Train all experts on the training split and perform Greedy Forward Selection
        based on validation performance.

        Args:
            max_train_samples (int, optional): If set, limits the number of training samples
                                               for faster debugging/iteration.
        """
        if self.data_train is None:
            self.load_data()

        # Apply subsampling if requested for the selection phase
        train_data_to_use = self.data_train
        if max_train_samples is not None:
            print(
                f"Subsampling training data to {max_train_samples} samples for selection phase..."
            )
            train_data_to_use = self._subsample_data(self.data_train, max_train_samples)

        # Delegate to SMFRIE_Trainer to fit experts and select the ensemble
        self.trainer.fit_and_select(train_data_to_use, self.data_val)

    def run_final_inference(self):
        """
        Phase 2: Retrain the selected experts on the combined (Train + Val) dataset
        and generate predictions for the Test set. Saves the submission file.
        """
        if self.data_train is None:
            self.load_data()

        # Check if selection has been run
        if not self.trainer.selected_experts:
            print(
                "Warning: No experts selected. Running selection phase with defaults..."
            )
            self.run_selection_phase()

        # Delegate to SMFRIE_Trainer for retraining and prediction
        # We use the full datasets here (no subsampling) for maximum performance
        final_probs = self.trainer.retrain_and_predict(
            self.data_train, self.data_val, self.data_test
        )

        # Format the submission
        submission_df = self.trainer.format_submission(
            final_probs, self.data_test["ids"]
        )

        # Save to disk
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved successfully to {SUBMISSION_PATH}")

        return submission_df
