import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import setup_logger, save_model, load_model
from library.modeling import ModelTrainer


class Trainer(ModelTrainer):
    """
    Orchestrates the training and submission generation for the
    Selective-Topology Orthogonal Manifold-Densified LDA solution.
    Inherits from ModelTrainer to reuse pipeline construction logic.
    """

    def __init__(self):
        super().__init__()
        self.logger = setup_logger("Trainer")

    def run_cross_validation(self, debug=False):
        """
        Executes the Stratified K-Fold Cross-Validation loop.

        Args:
            debug (bool): If True, runs only 2 folds for rapid testing.
        """
        self.logger.info(f"Starting Cross-Validation (Debug={debug})...")

        # 1. Setup Label Encoder
        # We must fit the encoder on ALL available species (Train + Val) to ensure
        # consistency across folds and handling of all classes.
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        all_species = pd.concat([df_train["species"], df_val["species"]]).unique()

        le = LabelEncoder()
        le.fit(all_species)
        save_model(le, "label_encoder.pkl")
        self.logger.info(f"LabelEncoder saved. Classes: {len(le.classes_)}")

        scores = []
        n_folds = 2 if debug else Config.N_FOLDS

        for fold_idx in range(n_folds):
            self.logger.info(f"--- Fold {fold_idx} ---")

            # 2. Get Data for the Fold
            # Data is already densified (3 centroids per image) by the DataManager
            train_data, train_labels, val_data, val_labels = (
                self.data_manager.get_fold_data(fold_idx, load_cached_data=True)
            )

            # 3. Prepare Features
            # _concat_features is inherited from ModelTrainer
            X_train, slices = self._concat_features(train_data)
            y_train = le.transform(train_labels)

            X_val, _ = self._concat_features(val_data)
            # Note: y_val is not transformed yet, we will handle it during aggregation

            # 4. Create and Train Pipeline
            # create_model_pipeline is inherited from ModelTrainer
            pipeline = self.create_model_pipeline(slices)
            pipeline.fit(X_train, y_train)

            # 5. Validation Inference (Full-Manifold Aggregation)
            # Predict on all 3 centroids of the validation set
            val_probs_densified = pipeline.predict_proba(X_val)

            # Cite debug_lesson_2: Align Model Predictions to the Full Label Space
            # In debug mode, the fold might not contain all classes.
            model_classes = pipeline.named_steps["classifier"].classes_
            if len(model_classes) < len(le.classes_):
                # Initialize full probability matrix with zeros
                full_probs = np.zeros((val_probs_densified.shape[0], len(le.classes_)))
                # Map partial predictions to the correct columns
                # Since y_train was encoded with 'le', model_classes are indices into le.classes_
                full_probs[:, model_classes] = val_probs_densified
                val_probs_densified = full_probs

            # To compute a representative metric, we must aggregate the 3 centroids per image
            # just like we do at test time.
            val_ids = val_data["ids"]

            # Create DataFrame for aggregation
            df_val_preds = pd.DataFrame(val_probs_densified, columns=le.classes_)
            df_val_preds["id"] = val_ids

            # Group by ID and compute mean probability vector
            df_val_agg = df_val_preds.groupby("id").mean()

            # Prepare Ground Truth for Aggregated IDs
            # Since all centroids of the same image have the same label, we drop duplicates
            df_labels = pd.DataFrame({"id": val_ids, "label": val_labels})
            df_labels = df_labels.drop_duplicates(subset=["id"]).set_index("id")

            # Align predictions and labels by ID
            common_ids = df_val_agg.index.intersection(df_labels.index)
            df_val_agg = df_val_agg.loc[common_ids]
            df_labels = df_labels.loc[common_ids]

            # Transform labels to integers
            y_true_agg = le.transform(df_labels["label"].values)
            y_pred_agg = df_val_agg.values

            # Clip probabilities to avoid log(0) extremes
            y_pred_agg = np.clip(y_pred_agg, 1e-15, 1 - 1e-15)

            # 6. Compute Metric
            # Cite debug_lesson_3: Explicitly Define Label Space for Multiclass Metrics on Sparse Subsets
            fold_score = log_loss(
                y_true_agg, y_pred_agg, labels=list(range(len(le.classes_)))
            )
            scores.append(fold_score)

            # Print full precision score
            self.logger.info(f"Fold {fold_idx} Aggregated Log Loss: {fold_score}")

            # 7. Save Model
            save_model(pipeline, f"pipeline_fold_{fold_idx}.pkl")

        # Summary
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        self.logger.info(f"CV Complete. Mean Log Loss: {mean_score}")
        self.logger.info(f"CV Std Log Loss: {std_score}")

    def generate_submission(self):
        """
        Generates predictions for the test set using the trained ensemble.
        Aggregates across folds and orthogonal centroids.
        """
        self.logger.info("Generating submission...")

        # 1. Load Test Data
        test_data, test_ids_densified = self.data_manager.get_test_data(
            load_cached_data=True
        )
        X_test, _ = self._concat_features(test_data)

        # 2. Load Label Encoder
        try:
            le = load_model("label_encoder.pkl")
        except FileNotFoundError:
            self.logger.error("LabelEncoder not found. Run cross-validation first.")
            return

        # 3. Ensemble Inference
        fold_probs = []
        for fold_idx in range(Config.N_FOLDS):
            try:
                pipeline = load_model(f"pipeline_fold_{fold_idx}.pkl")
                probs = pipeline.predict_proba(X_test)
                fold_probs.append(probs)
            except FileNotFoundError:
                self.logger.warning(f"Model for fold {fold_idx} not found. Skipping.")
                continue

        if not fold_probs:
            raise RuntimeError("No trained models found. Cannot generate submission.")

        # Average probabilities across all folds
        avg_fold_probs = np.mean(fold_probs, axis=0)

        # 4. Centroid Aggregation
        # The test set is densified (3 centroids per image). We must average them.
        df_preds = pd.DataFrame(avg_fold_probs, columns=le.classes_)
        df_preds["id"] = test_ids_densified

        # Group by ID and mean
        final_preds = df_preds.groupby("id").mean().reset_index()

        # 5. Post-Processing
        # Clip probabilities to valid range [0, 1] and avoid extremes
        numeric_cols = final_preds.columns.drop("id")
        final_preds[numeric_cols] = final_preds[numeric_cols].clip(1e-15, 1 - 1e-15)

        # 6. Save Submission
        final_preds.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved successfully to {Config.SUBMISSION_PATH}")

        # Print sample for verification
        self.logger.info("Sample predictions:")
        print(final_preds.head())
