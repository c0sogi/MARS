import os
import pickle
import numpy as np
import pandas as pd
from library.config import Config
from library.data_processing import DatasetManager


class InferenceEngine:
    """
    Manages the inference process using the ensemble of trained models.
    Performs Full-Manifold Test-Time Aggregation (TTA) by predicting on
    3 orthogonal centroids per test image and averaging the results.
    """

    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")

    def run(self, load_cached_data=True):
        """
        Executes the inference pipeline:
        1. Loads test data (features).
        2. Prepares densified test set (3 centroids per image).
        3. Loads class definitions.
        4. Aggregates predictions from all K-Fold models.
        5. Saves submission file.

        Args:
            load_cached_data (bool): Whether to use cached features.
        """
        print("Starting Inference Engine...")

        # 1. Load Data
        # We only need the test set here, but load_data returns both structure
        data = self.dataset_manager.load_data(load_cached_data=load_cached_data)
        test_data_dict = data["test"]

        # 2. Prepare Densified Test Data
        # Converts N images -> 3N samples (3 orthogonal centroids per image)
        print("Preparing densified test set...")
        X_test, test_ids_expanded = self.dataset_manager.prepare_training_set(
            test_data_dict
        )

        # Recover unique IDs (every 3rd element) to match the aggregated predictions
        # test_ids_expanded is [id1, id1, id1, id2, id2, id2...]
        test_ids = test_ids_expanded[::3]
        n_test = len(test_ids)

        # 3. Load Class Definitions
        classes_path = os.path.join(self.models_dir, "classes.pkl")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(
                f"Classes file not found at {classes_path}. "
                "Ensure training has been completed successfully."
            )

        with open(classes_path, "rb") as f:
            classes = pickle.load(f)

        n_classes = len(classes)

        # Accumulator for ensemble predictions
        ensemble_probs = np.zeros((n_test, n_classes))
        models_found = 0

        # 4. Ensemble Prediction Loop
        print(f"Aggregating predictions from {Config.N_FOLDS} folds...")

        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(self.models_dir, f"pipeline_fold_{fold}.pkl")

            if not os.path.exists(model_path):
                print(
                    f"Warning: Model for fold {fold} not found at {model_path}. Skipping."
                )
                continue

            try:
                with open(model_path, "rb") as f:
                    pipeline = pickle.load(f)

                # Predict on densified test set (3N samples)
                # Shape: (3 * N_test, n_classes)
                probs_expanded = pipeline.predict_proba(X_test)

                # Reshape and Average Centroids (TTA)
                # We average the predictions of the 3 views for each model
                # Shape: (N_test, 3, n_classes) -> (N_test, n_classes)
                probs_reshaped = probs_expanded.reshape(n_test, 3, n_classes)
                probs_mean = np.mean(probs_reshaped, axis=1)

                ensemble_probs += probs_mean
                models_found += 1

            except Exception as e:
                print(f"Error loading or predicting with fold {fold}: {e}")

        if models_found == 0:
            raise RuntimeError(
                "No models were successfully loaded. Cannot generate submission."
            )

        # Average across folds
        ensemble_probs /= models_found
        print(f"Ensemble aggregation complete using {models_found} models.")

        # 5. Generate Submission File
        df_sub = pd.DataFrame(ensemble_probs, columns=classes)
        df_sub.insert(0, "id", test_ids)

        Config.make_dirs()  # Ensure submission directory exists
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
