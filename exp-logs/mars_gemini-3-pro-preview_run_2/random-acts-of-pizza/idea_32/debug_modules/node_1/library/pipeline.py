import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin

from library.config import Config
from library.data_loader import PizzaDataLoader
from library.feature_extraction import FeatureExtractor
from library.preprocessing import MultiModalTransformer
from library.model import ModelFactory


class DictSliceTransformer(BaseEstimator, TransformerMixin):
    """
    Helper transformer to allow GridSearchCV to work with a dictionary of arrays.
    It takes indices as input (X) and returns the sliced dictionary from the stored dataset.
    """

    def __init__(self, data_dict):
        self.data_dict = data_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is expected to be an array of indices
        indices = X
        sliced = {}
        for k, v in self.data_dict.items():
            if isinstance(v, pd.DataFrame):
                sliced[k] = v.iloc[indices].reset_index(drop=True)
            else:
                sliced[k] = v[indices]
        return sliced


class CrossValidationManager:
    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.submission_dir = Config.SUBMISSION_DIR
        self.loader = PizzaDataLoader()
        self.extractor = FeatureExtractor()

        # Ensure directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

    def _slice_dict(self, data_dict, indices):
        """Helper to slice the data dictionary based on indices."""
        sliced = {}
        for k, v in data_dict.items():
            if isinstance(v, pd.DataFrame):
                sliced[k] = v.iloc[indices].reset_index(drop=True)
            else:
                sliced[k] = v[indices]
        return sliced

    def run_cv(self, load_cached_data=True):
        print("Starting 5-Fold Stratified Cross-Validation...")

        # 1. Load Data (Train + Val merged)
        df_train = self.loader.load_data("train", load_cached_data=load_cached_data)
        df_val = self.loader.load_data("val", load_cached_data=load_cached_data)

        # Concatenate
        df_combined = pd.concat([df_train, df_val], ignore_index=True)
        y_combined = df_combined["requester_received_pizza"].values

        # 2. Extract Features
        # Extract separately to utilize caching logic per split
        feats_train = self.extractor.extract_features(
            df_train, "train", load_cached_data=load_cached_data
        )
        feats_val = self.extractor.extract_features(
            df_val, "val", load_cached_data=load_cached_data
        )

        # Merge feature dictionaries
        X_combined = {}
        keys = ["anchor", "semantic_aux", "affective_aux"]
        for key in keys:
            X_combined[key] = np.vstack([feats_train[key], feats_val[key]])

        # 3. Extract Metadata
        meta_train = self.loader.get_metadata_features(df_train)
        meta_val = self.loader.get_metadata_features(df_val)
        X_combined["metadata"] = pd.concat([meta_train, meta_val], ignore_index=True)

        # 4. Prepare for CV
        skf = StratifiedKFold(
            n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
        )
        indices = np.arange(len(y_combined))

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(indices, y_combined)):
            print(f"\n--- Fold {fold} ---")

            # Create indices for this fold
            X_train_indices = indices[train_idx]
            y_train = y_combined[train_idx]

            # Prepare Validation Data (Sliced)
            X_val_dict = self._slice_dict(X_combined, val_idx)
            y_val = y_combined[val_idx]

            # --- Hyperparameter Tuning ---
            # Construct Search Pipeline: IndexLoader -> Preprocessor -> Classifier
            # We pass the FULL X_combined to the loader, but it will only slice what it receives in fit/transform (train_idx)
            search_loader = DictSliceTransformer(X_combined)
            search_pipeline = Pipeline(
                [
                    ("loader", search_loader),
                    ("prep", MultiModalTransformer()),
                    ("bagging", ModelFactory.get_classifier()),
                ]
            )

            param_grid = ModelFactory.get_hyperparameter_grid()

            # Grid Search (Internal CV)
            # We pass X_train_indices. The pipeline's first step 'loader' will use these to slice X_combined.
            grid = GridSearchCV(
                search_pipeline,
                param_grid,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=0,
            )

            print("Running Grid Search...")
            grid.fit(X_train_indices, y_train)

            best_params = grid.best_params_
            print(f"Best Parameters: {best_params}")
            print(f"Best Internal CV Score: {grid.best_score_}")

            # --- Final Training for Fold ---
            # Construct Clean Pipeline (No Loader) to save memory and decouple from dataset
            clean_pipeline = Pipeline(
                [
                    ("prep", MultiModalTransformer()),
                    ("bagging", ModelFactory.get_classifier()),
                ]
            )

            # Set params
            clean_pipeline.set_params(**best_params)

            # Slice Training Data Manually
            X_train_dict = self._slice_dict(X_combined, train_idx)

            # Fit Clean Pipeline
            print("Fitting final model for fold...")
            clean_pipeline.fit(X_train_dict, y_train)

            # --- Evaluation ---
            y_pred_proba = clean_pipeline.predict_proba(X_val_dict)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            fold_scores.append(auc)
            print(f"Fold {fold} Validation AUC: {auc}")

            # --- Save Model ---
            model_path = os.path.join(self.working_dir, f"fold_{fold}_pipeline.joblib")
            joblib.dump(clean_pipeline, model_path)
            print(f"Saved model to {model_path}")

        print("\n==============================")
        print(f"Mean CV AUC: {np.mean(fold_scores)}")
        print(f"Std CV AUC: {np.std(fold_scores)}")
        print("==============================")

    def generate_submission(self, load_cached_data=True):
        print("\nGenerating Submission...")

        # 1. Load Test Data
        df_test = self.loader.load_data("test", load_cached_data=load_cached_data)

        # 2. Extract Features
        feats_test = self.extractor.extract_features(
            df_test, "test", load_cached_data=load_cached_data
        )
        meta_test = self.loader.get_metadata_features(df_test)

        # Combine into X_test dict
        X_test = {**feats_test, "metadata": meta_test}

        # 3. Load Models and Predict
        fold_predictions = []

        for fold in range(Config.NUM_FOLDS):
            model_path = os.path.join(self.working_dir, f"fold_{fold}_pipeline.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            print(f"Loading model from {model_path}...")
            pipeline = joblib.load(model_path)

            # Predict
            preds = pipeline.predict_proba(X_test)[:, 1]
            fold_predictions.append(preds)

        # 4. Average Predictions (Bagging)
        avg_preds = np.mean(fold_predictions, axis=0)

        # 5. Create Submission File
        submission_df = pd.DataFrame(
            {"request_id": df_test["request_id"], "requester_received_pizza": avg_preds}
        )

        submission_path = os.path.join(self.submission_dir, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
