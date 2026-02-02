import os
import joblib
import numpy as np
import pandas as pd
from library.config import Config
from library.densification import get_densified_data

# Must import custom transformer for joblib to deserialize pipelines correctly
from library.custom_transformers import DualStreamPreprocessor


class InferenceManager:
    """
    Manages the inference process for the Leaf Classification task.
    Loads the trained ensemble, generates predictions on the test set,
    aggregates results across views and folds, and formats the submission.
    """

    def __init__(self):
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        self.submission_path = Config.SUBMISSION_PATH
        self.sample_submission_path = os.path.join(
            Config.INPUT_DIR, "sample_submission.csv"
        )
        self.n_folds = Config.N_FOLDS

    def generate_submission(self, load_cached_data=True):
        """
        Generates the submission file by running the ensemble on the test set.

        Args:
            load_cached_data (bool): Whether to load test features from cache.
        """
        print("Initializing inference pipeline...")

        # 1. Load Densified Test Data
        # Returns 3 samples per image (Centroids A, B, C)
        ids, X_dino, X_conv, X_tab, _ = get_densified_data(
            split="test", load_cached_data=load_cached_data
        )

        # 2. Concatenate Features
        # The pipeline expects [DINO, ConvNeXt, Tabular]
        print(
            f"Test data shapes - DINO: {X_dino.shape}, ConvNeXt: {X_conv.shape}, Tabular: {X_tab.shape}"
        )
        X = np.concatenate([X_dino, X_conv, X_tab], axis=1)

        # 3. Load Sample Submission for Column Ordering
        if not os.path.exists(self.sample_submission_path):
            raise FileNotFoundError(
                f"Sample submission not found at {self.sample_submission_path}"
            )

        sample_sub_df = pd.read_csv(self.sample_submission_path)
        required_columns = sample_sub_df.columns.tolist()

        # 4. Ensemble Prediction Loop
        ensemble_df = None
        successful_folds = 0

        print(f"Starting inference with {self.n_folds} folds...")

        for fold in range(self.n_folds):
            model_path = os.path.join(self.models_dir, f"pipeline_fold_{fold}.pkl")

            if not os.path.exists(model_path):
                print(
                    f"Warning: Model for fold {fold} not found at {model_path}. Skipping."
                )
                continue

            # Load Pipeline
            # DualStreamPreprocessor is needed in namespace for this to work
            try:
                pipeline = joblib.load(model_path)
            except Exception as e:
                print(f"Error loading model fold {fold}: {e}")
                continue

            # Predict Probabilities
            # Shape: (3 * N_images, N_classes)
            # Pipeline handles scaling/transforming internally
            probas = pipeline.predict_proba(X)

            # Create DataFrame for this fold's predictions
            # Columns are the class names stored in the model
            fold_df = pd.DataFrame(probas, columns=pipeline.classes_)
            fold_df["id"] = ids

            # Aggregate Views (Centroids A, B, C) -> Mean per Image
            # Group by ID and calculate mean probability vector
            fold_agg = fold_df.groupby("id").mean()

            # Accumulate
            if ensemble_df is None:
                ensemble_df = fold_agg
            else:
                # Add to running total (aligns by index 'id' automatically)
                ensemble_df = ensemble_df.add(fold_agg, fill_value=0)

            successful_folds += 1

        if successful_folds == 0:
            raise RuntimeError(
                "No models were successfully loaded. Cannot generate submission."
            )

        print(f"Aggregated predictions from {successful_folds} models.")

        # 5. Average across folds
        ensemble_df /= successful_folds

        # 6. Format Submission
        # Reset index to make 'id' a column again
        ensemble_df = ensemble_df.reset_index()

        # Ensure all required columns exist
        # (Model classes should match sample submission classes, but we verify)
        pred_cols = set(ensemble_df.columns)
        req_cols_set = set(required_columns)

        missing = req_cols_set - pred_cols
        if missing:
            print(
                f"Warning: Filling {len(missing)} missing columns with 0.0 (Debug/Subsample mode detected)."
            )
            # Create a DataFrame with missing columns set to 0.0
            missing_df = pd.DataFrame(
                0.0, index=ensemble_df.index, columns=list(missing)
            )
            # Concatenate horizontally
            ensemble_df = pd.concat([ensemble_df, missing_df], axis=1)

        # Reorder columns to match sample_submission exactly
        submission_df = ensemble_df[required_columns]

        # 7. Save
        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)
        submission_df.to_csv(self.submission_path, index=False)
        print(f"Submission saved to {self.submission_path}")
        print("Inference complete.")
