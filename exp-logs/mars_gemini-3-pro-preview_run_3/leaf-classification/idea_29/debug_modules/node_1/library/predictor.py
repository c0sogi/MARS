import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logger, load_model
from library.modeling import ModelTrainer


class Predictor(ModelTrainer):
    """
    Handles the generation of predictions for the test set using the trained
    Selective-Topology Orthogonal Manifold-Densified LDA ensemble.
    """

    def __init__(self):
        # Initialize parent to setup data_manager and logger
        super().__init__()
        self.logger = setup_logger("Predictor")

    def generate_submission(self):
        """
        Generates the submission file by:
        1. Loading densified test data (3 centroids per image).
        2. Loading the LabelEncoder and trained Ensemble models.
        3. Predicting probabilities using all models.
        4. Aggregating predictions across folds (mean).
        5. Aggregating predictions across orthogonal centroids (mean).
        6. Clipping probabilities and saving to CSV.
        """
        self.logger.info("Starting submission generation process...")

        # 1. Load Test Data
        # The data manager handles caching of the feature extraction and densification
        try:
            test_data, test_ids_densified = self.data_manager.get_test_data(
                load_cached_data=True
            )
        except Exception as e:
            self.logger.error(f"Failed to load test data: {e}")
            raise e

        # Prepare input matrix X using the inherited helper
        # This handles the concatenation of dino, conv, and tab features
        X_test, _ = self._concat_features(test_data)
        self.logger.info(f"Test data shape: {X_test.shape}")

        # 2. Load Label Encoder
        try:
            le = load_model("label_encoder.pkl")
            class_names = le.classes_
            self.logger.info(f"Loaded LabelEncoder with {len(class_names)} classes.")
        except FileNotFoundError:
            self.logger.error("LabelEncoder not found. Ensure training has been run.")
            raise

        # 3. Ensemble Inference
        fold_probs = []
        successful_models = 0

        for fold_idx in range(Config.N_FOLDS):
            model_filename = f"pipeline_fold_{fold_idx}.pkl"
            try:
                pipeline = load_model(model_filename)

                # Predict probabilities
                # Output shape: (N_test_densified, N_classes)
                probs = pipeline.predict_proba(X_test)
                fold_probs.append(probs)
                successful_models += 1

            except FileNotFoundError:
                self.logger.warning(f"Model {model_filename} not found. Skipping fold.")
                continue
            except Exception as e:
                self.logger.warning(f"Error predicting with {model_filename}: {e}")
                continue

        if successful_models == 0:
            raise RuntimeError(
                "No trained models could be loaded. Cannot generate submission."
            )

        self.logger.info(f"Aggregating predictions from {successful_models} models...")

        # 4. Average probabilities across all folds
        # Shape: (N_test_densified, N_classes)
        avg_fold_probs = np.mean(fold_probs, axis=0)

        # 5. Centroid Aggregation (Full-Manifold Aggregation)
        # The test set is densified (3 centroids per image).
        # We must average the predictions for these centroids to get one prediction per image ID.

        # Create a temporary DataFrame to facilitate grouping
        df_preds = pd.DataFrame(avg_fold_probs, columns=class_names)
        df_preds["id"] = test_ids_densified

        # Group by 'id' and compute the mean of probabilities
        final_preds = df_preds.groupby("id").mean().reset_index()

        self.logger.info(f"Aggregated shapes: {final_preds.shape} (Rows, Columns)")

        # 6. Post-Processing
        # Clip probabilities to valid range [1e-15, 1-1e-15] to avoid log(0) penalties
        # Identify numeric columns (all except 'id')
        numeric_cols = final_preds.columns.drop("id")
        final_preds[numeric_cols] = final_preds[numeric_cols].clip(1e-15, 1 - 1e-15)

        # 7. Save Submission
        self.logger.info(f"Saving submission to {Config.SUBMISSION_PATH}...")
        final_preds.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info("Submission saved successfully.")
