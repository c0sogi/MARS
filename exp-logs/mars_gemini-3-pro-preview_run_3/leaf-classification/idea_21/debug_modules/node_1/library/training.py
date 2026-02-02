import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from library.config import Config
from library.utils import setup_logger, seed_everything
from library.data_processing import OrthogonalDataManager
from library.pipeline import create_expert_pipeline


class OSLDETrainer:
    """
    Manages the training, evaluation, and submission generation for the
    Orientation-Specialized Linear Discriminant Experts (OS-LDE) system.
    """

    def __init__(self):
        self.logger = setup_logger("OSLDETrainer")
        self.data_manager = OrthogonalDataManager()
        self.label_encoder = LabelEncoder()

    def _prepare_input(self, centroid_data, tabular_data, indices=None):
        """
        Concatenates image centroids and tabular data to form the full feature vector.

        Args:
            centroid_data (np.ndarray): Image features for a specific orthogonal set (N, 2560).
            tabular_data (np.ndarray): Tabular features (N, 192).
            indices (np.ndarray, optional): Indices to subset the data (e.g., for train/val split).

        Returns:
            np.ndarray: Concatenated feature matrix.
        """
        if indices is not None:
            c_data = centroid_data[indices]
            t_data = tabular_data[indices]
        else:
            c_data = centroid_data
            t_data = tabular_data

        return np.hstack([c_data, t_data])

    def train_and_evaluate(self):
        """
        Executes the Stratified K-Fold training loop.
        Trains 3 experts (A, B, C) per fold, evaluates on validation data,
        and saves the trained pipelines.

        Returns:
            float: Average Log Loss across all folds.
        """
        seed_everything(Config.SEED)

        # 1. Load Training Data
        self.logger.info("Loading training data...")
        train_data = self.data_manager.get_data("train", load_cached_data=True)

        centroids = train_data["centroids"]  # Dict {'A': ..., 'B': ..., 'C': ...}
        tabular = train_data["tabular"]
        labels = train_data["labels"]
        # ids = train_data["ids"] # Not needed for training logic

        # Encode Labels
        y_encoded = self.label_encoder.fit_transform(labels)
        classes = self.label_encoder.classes_

        # Save Label Encoder for inference usage
        le_path = os.path.join(Config.WORKING_DIR, "label_encoder.pkl")
        joblib.dump(self.label_encoder, le_path)

        # 2. Get Folds
        folds = self.data_manager.get_folds(y_encoded)

        oof_preds = np.zeros((len(labels), len(classes)))
        fold_scores = []

        # 3. Training Loop
        for fold, (train_idx, val_idx) in enumerate(folds):
            self.logger.info(f"Starting Fold {fold + 1}/{Config.N_FOLDS}")

            # Prepare Targets
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            fold_probs = np.zeros((len(val_idx), len(classes)))

            # Train each expert (A, B, C) independently
            for expert_name in ["A", "B", "C"]:
                # Prepare Features: [Centroid_X | Tabular]
                X_train = self._prepare_input(
                    centroids[expert_name], tabular, train_idx
                )
                X_val = self._prepare_input(centroids[expert_name], tabular, val_idx)

                # Create Pipeline
                pipeline = create_expert_pipeline()

                # Fit
                pipeline.fit(X_train, y_train)

                # Predict
                probs = pipeline.predict_proba(X_val)

                # Accumulate for Ensemble Averaging
                fold_probs += probs

                # Save Model
                model_filename = f"model_fold_{fold}_expert_{expert_name}.pkl"
                model_path = os.path.join(Config.WORKING_DIR, model_filename)
                joblib.dump(pipeline, model_path)

            # Average Predictions (Ensemble of 3 experts)
            fold_probs /= 3.0

            # Store OOF predictions
            oof_preds[val_idx] = fold_probs

            # Compute Fold Metric
            # Clip probabilities to avoid log(0) extremes
            fold_probs_clipped = np.clip(
                fold_probs, Config.PROB_CLIP_MIN, Config.PROB_CLIP_MAX
            )
            # Re-normalize rows to sum to 1
            fold_probs_clipped /= fold_probs_clipped.sum(axis=1, keepdims=True)

            score = log_loss(
                y_val, fold_probs_clipped, labels=list(range(len(classes)))
            )
            fold_scores.append(score)

            self.logger.info(f"Fold {fold + 1} Log Loss: {score:.15f}")

        # 4. Overall Evaluation
        avg_score = np.mean(fold_scores)
        self.logger.info(
            f"Average Log Loss across {Config.N_FOLDS} folds: {avg_score:.15f}"
        )

        # Calculate OOF Score
        oof_preds_clipped = np.clip(
            oof_preds, Config.PROB_CLIP_MIN, Config.PROB_CLIP_MAX
        )
        oof_preds_clipped /= oof_preds_clipped.sum(axis=1, keepdims=True)
        oof_score = log_loss(
            y_encoded, oof_preds_clipped, labels=list(range(len(classes)))
        )
        self.logger.info(f"OOF Log Loss: {oof_score:.15f}")

        return avg_score

    def generate_submission(self):
        """
        Loads test data and all trained models to generate predictions.
        Averages predictions across all folds and all experts.
        Saves the result to submission.csv.
        """
        self.logger.info("Starting submission generation...")

        # 1. Load Test Data
        test_data = self.data_manager.get_data("test", load_cached_data=True)
        centroids = test_data["centroids"]
        tabular = test_data["tabular"]
        test_ids = test_data["ids"]

        # Load Label Encoder
        le_path = os.path.join(Config.WORKING_DIR, "label_encoder.pkl")
        if not os.path.exists(le_path):
            raise FileNotFoundError("Label encoder not found. Run training first.")
        label_encoder = joblib.load(le_path)
        classes = label_encoder.classes_

        # Initialize predictions accumulator
        final_preds = np.zeros((len(test_ids), len(classes)))

        # 2. Iterate through Folds and Experts
        # We aggregate predictions from (N_FOLDS * 3) models
        models_found = 0

        for fold in range(Config.N_FOLDS):
            fold_preds = np.zeros((len(test_ids), len(classes)))

            for expert_name in ["A", "B", "C"]:
                # Prepare Input for this expert
                X_test = self._prepare_input(centroids[expert_name], tabular)

                # Load Model
                model_filename = f"model_fold_{fold}_expert_{expert_name}.pkl"
                model_path = os.path.join(Config.WORKING_DIR, model_filename)

                if not os.path.exists(model_path):
                    self.logger.warning(f"Model {model_filename} not found. Skipping.")
                    continue

                pipeline = joblib.load(model_path)

                # Predict
                probs = pipeline.predict_proba(X_test)
                fold_preds += probs
                models_found += 1

            # Average experts for this fold (divide by 3)
            fold_preds /= 3.0

            # Add to final ensemble
            final_preds += fold_preds

        if models_found == 0:
            raise RuntimeError("No trained models found to generate submission.")

        # Average across folds
        final_preds /= Config.N_FOLDS

        # 3. Post-processing
        # Clip probabilities
        final_preds = np.clip(final_preds, Config.PROB_CLIP_MIN, Config.PROB_CLIP_MAX)
        # Normalize
        final_preds /= final_preds.sum(axis=1, keepdims=True)

        # 4. Create Submission DataFrame
        df_sub = pd.DataFrame(final_preds, columns=classes)
        df_sub.insert(0, "id", test_ids.astype(int))

        # Save
        self.logger.info(f"Saving submission to {Config.SUBMISSION_PATH}")
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info("Submission generation complete.")
