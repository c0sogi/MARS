import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.data_loader import HistogramDataLoader
from library.model import BirdRandomForest


class Trainer:
    """
    Orchestrates training, evaluation, and submission generation for the Bird Species Classification task.
    """

    def __init__(self):
        """
        Initialize the Trainer with configuration, data loader, and model.
        """
        self.config = Config
        self.loader = HistogramDataLoader()
        self.model = BirdRandomForest()
        self.num_species = self.config.NUM_SPECIES

    def train(self, load_cached_data=True, max_samples=None):
        """
        Trains the model and evaluates it on the validation set.

        Args:
            load_cached_data (bool): Whether to use cached preprocessed data.
            max_samples (int, optional): Limit the number of samples for debugging.

        Returns:
            float: The macro-averaged ROC AUC score on the validation set.
        """
        print("Retrieving data splits...")
        (X_train, y_train), (X_val, y_val), _ = self.loader.get_data_splits(
            load_cached_data=load_cached_data, max_samples=max_samples
        )

        print(
            f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features..."
        )
        self.model.fit(X_train, y_train)
        print("Training complete.")

        print("Evaluating on validation set...")
        y_val_pred = self.model.predict_proba(X_val)

        # Calculate ROC AUC
        # We calculate per-class AUC to handle cases where a class might be missing in validation
        auc_scores = []
        for i in range(self.num_species):
            # Check if the class is present in y_val
            if len(np.unique(y_val[:, i])) > 1:
                score = roc_auc_score(y_val[:, i], y_val_pred[:, i])
                auc_scores.append(score)
            else:
                # If only one class is present (e.g. all 0s), AUC is undefined.
                # We skip it or treat it as 0.5 (random guess) depending on preference.
                # Here we skip to avoid skewing the average with undefined math.
                pass

        if auc_scores:
            mean_auc = np.mean(auc_scores)
            print(f"Validation ROC AUC (Macro): {mean_auc}")
        else:
            mean_auc = 0.0
            print("Validation ROC AUC: Undefined (no valid classes in validation set)")

        return mean_auc

    def generate_submission(self, load_cached_data=True, max_samples=None):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            load_cached_data (bool): Whether to use cached preprocessed data.
            max_samples (int, optional): Limit the number of samples for debugging.
        """
        print("Retrieving test data...")
        _, _, (X_test, test_ids) = self.loader.get_data_splits(
            load_cached_data=load_cached_data, max_samples=max_samples
        )

        print(f"Predicting on {X_test.shape[0]} test samples...")
        # Get probabilities: shape (n_samples, n_species)
        y_test_pred = self.model.predict_proba(X_test)

        # Prepare submission data
        # Format: Id, Probability
        # Id = rec_id * 100 + species_id

        submission_rows = []

        # Iterate through each recording and each species to flatten the results
        for idx, rec_id in enumerate(test_ids):
            probs = y_test_pred[idx]
            for species_id in range(self.num_species):
                row_id = int(rec_id * 100 + species_id)
                probability = probs[species_id]
                submission_rows.append({"Id": row_id, "Probability": probability})

        df_submission = pd.DataFrame(submission_rows)

        # Sort by Id just to be neat (though not strictly required if all IDs are present)
        df_submission = df_submission.sort_values("Id")

        # Save to file
        output_path = self.config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        print(f"Submission shape: {df_submission.shape}")
