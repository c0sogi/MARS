import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from library.config import (
    NUM_FOLDS,
    RANDOM_SEED,
    SUBMISSION_PATH,
    ID_COL,
    TARGET_COL,
)
from library.model_definitions import ModelFactory


class EnsembleManager:
    """
    Orchestrates the training, validation, and inference of the Stacking Ensemble.
    Implements the Validation-Guided Retraining protocol.
    """

    def __init__(self):
        self.level_1_models = ModelFactory.get_level_1_models()
        self.meta_learner = ModelFactory.get_level_2_model()
        self.oof_predictions = None
        self.test_predictions = None

    def _slice_features(self, feature_dict, indices):
        """
        Slices a dictionary of feature arrays based on provided indices.
        Handles both numpy arrays and sparse matrices.
        """
        sliced = {}
        for key, data in feature_dict.items():
            if sparse.issparse(data):
                sliced[key] = data[indices]
            else:
                sliced[key] = data[indices]
        return sliced

    def _concat_features(self, dict1, dict2):
        """
        Concatenates two feature dictionaries along the 0-axis (rows).
        """
        concatenated = {}
        for key in dict1.keys():
            d1 = dict1[key]
            d2 = dict2[key]

            if sparse.issparse(d1):
                concatenated[key] = sparse.vstack([d1, d2]).tocsr()
            else:
                concatenated[key] = np.vstack([d1, d2])
        return concatenated

    def generate_oof_predictions(self, X_train_df, y_train, feature_pipeline):
        """
        Performs Stratified K-Fold Cross-Validation to generate Out-of-Fold (OOF) predictions
        for all Level 1 base learners.

        Args:
            X_train_df (pd.DataFrame): Training data (raw).
            y_train (pd.Series): Training targets.
            feature_pipeline (FeaturePipeline): Fitted feature pipeline.

        Returns:
            pd.DataFrame: OOF predictions for each base learner.
        """
        print(f"Starting {NUM_FOLDS}-Fold Cross-Validation for OOF generation...")

        # 1. Transform entire training set once
        train_feats = feature_pipeline.transform(X_train_df, prefix="train")

        # 2. Initialize OOF storage
        n_samples = len(y_train)
        model_keys = list(self.level_1_models.keys())
        oof_preds = pd.DataFrame(0.0, index=y_train.index, columns=model_keys)

        # 3. CV Loop
        skf = StratifiedKFold(
            n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED
        )

        # Convert y_train to numpy for indexing
        y_train_np = y_train.values
        indices = np.arange(n_samples)

        for fold, (train_idx, val_idx) in enumerate(skf.split(indices, y_train_np)):
            print(f"  Processing Fold {fold + 1}/{NUM_FOLDS}...")

            # Slice features for this fold
            X_fold_train_dict = self._slice_features(train_feats, train_idx)
            X_fold_val_dict = self._slice_features(train_feats, val_idx)
            y_fold_train = y_train_np[train_idx]
            # y_fold_val = y_train_np[val_idx] # Not strictly needed for training unless early stopping used in CV

            # Train each base learner
            for key, model in self.level_1_models.items():
                # Prepare specific feature matrix for this model
                X_model_train = ModelFactory.prepare_features(X_fold_train_dict, key)
                X_model_val = ModelFactory.prepare_features(X_fold_val_dict, key)

                # Clone model (re-instantiate) to ensure fresh start
                # We use the factory again or clone. Factory is safer given the structure.
                fresh_models = ModelFactory.get_level_1_models()
                current_model = fresh_models[key]

                # Fit
                # Note: We do NOT use early stopping in CV loop for simplicity and consistency with standard stacking,
                # unless explicitly required. The prompt emphasizes early stopping for the FINAL retraining.
                # For XGBoost in CV, we use the n_estimators defined in config (which might be high),
                # but without a validation set passed to fit(), it runs to completion.
                # To prevent excessive runtime, we rely on the config parameters being reasonable.
                current_model.fit(X_model_train, y_fold_train)

                # Predict (Probabilities)
                if hasattr(current_model, "predict_proba"):
                    probs = current_model.predict_proba(X_model_val)[:, 1]
                else:
                    # Fallback for models without predict_proba (unlikely here)
                    probs = current_model.predict(X_model_val)

                # Store OOF
                # Map numpy indices back to DataFrame index
                original_indices = y_train.index[val_idx]
                oof_preds.loc[original_indices, key] = probs

        # 4. Evaluate OOF Performance
        print("\n--- OOF Performance (AUC) ---")
        for key in model_keys:
            auc = roc_auc_score(y_train, oof_preds[key])
            print(f"{key}: {auc}")

        self.oof_predictions = oof_preds
        return oof_preds

    def train_meta_learner(self, y_train):
        """
        Trains the Level 2 Meta-Learner (Logistic Regression) on the OOF predictions.

        Args:
            y_train (pd.Series): Training targets.
        """
        print("\nTraining Meta-Learner...")
        if self.oof_predictions is None:
            raise ValueError(
                "OOF predictions not generated. Run generate_oof_predictions first."
            )

        self.meta_learner.fit(self.oof_predictions, y_train)

        # Print coefficients
        coefs = self.meta_learner.coef_[0]
        intercept = self.meta_learner.intercept_[0]
        feature_names = self.oof_predictions.columns

        print("Meta-Learner Coefficients:")
        for name, coef in zip(feature_names, coefs):
            print(f"  {name}: {coef}")
        print(f"  Intercept: {intercept}")

        # Evaluate Meta-Learner on OOF (Optimistic estimate)
        meta_oof_preds = self.meta_learner.predict_proba(self.oof_predictions)[:, 1]
        auc = roc_auc_score(y_train, meta_oof_preds)
        print(f"Meta-Learner OOF AUC: {auc}")

    def retrain_final_models(
        self, X_train_df, y_train, X_val_df, y_val, feature_pipeline
    ):
        """
        Retrains the Level 1 models on the full dataset using the Validation-Guided protocol.

        Protocol:
        - RF / Linear: Train on Concatenated (Train + Val).
        - XGBoost: Train on Train, use Val for Early Stopping.
        """
        print("\nRetraining Final Level 1 Models...")

        # 1. Transform Data
        train_feats = feature_pipeline.transform(X_train_df, prefix="train")
        val_feats = feature_pipeline.transform(X_val_df, prefix="val")

        # 2. Prepare Concatenated Data (for RF/Linear)
        full_feats = self._concat_features(train_feats, val_feats)
        y_full = pd.concat([y_train, y_val])

        # 3. Retrain Loop
        for key, model in self.level_1_models.items():
            print(f"  Retraining {key}...")

            # Determine strategy based on model type
            is_xgb = "xgb" in key

            if is_xgb:
                # XGBoost Strategy: Train on Train, Early Stop on Val
                X_train_m = ModelFactory.prepare_features(train_feats, key)
                X_val_m = ModelFactory.prepare_features(val_feats, key)

                fit_params = ModelFactory.get_xgb_fit_params()

                model.fit(
                    X_train_m,
                    y_train,
                    eval_set=[(X_val_m, y_val)],
                    verbose=False,
                    **fit_params,
                )
            else:
                # RF/Linear Strategy: Train on Full (Train + Val)
                X_full_m = ModelFactory.prepare_features(full_feats, key)
                model.fit(X_full_m, y_full)

    def predict_test(self, X_test_df, feature_pipeline):
        """
        Generates predictions for the test set using the stacked ensemble.

        Args:
            X_test_df (pd.DataFrame): Test data.
            feature_pipeline (FeaturePipeline): Fitted feature pipeline.

        Returns:
            pd.DataFrame: Submission dataframe.
        """
        print("\nGenerating Test Predictions...")

        # 1. Transform Test Data
        test_feats = feature_pipeline.transform(X_test_df, prefix="test")

        # 2. Generate Level 1 Predictions
        level_1_test_preds = pd.DataFrame(
            index=X_test_df.index, columns=self.level_1_models.keys()
        )

        for key, model in self.level_1_models.items():
            X_test_m = ModelFactory.prepare_features(test_feats, key)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test_m)[:, 1]
            else:
                probs = model.predict(X_test_m)
            level_1_test_preds[key] = probs

        # 3. Generate Meta Prediction
        final_probs = self.meta_learner.predict_proba(level_1_test_preds)[:, 1]

        # 4. Create Submission DataFrame
        submission = pd.DataFrame({ID_COL: X_test_df[ID_COL], TARGET_COL: final_probs})

        # Save
        print(f"Saving submission to {SUBMISSION_PATH}...")
        submission.to_csv(SUBMISSION_PATH, index=False)

        return submission

    def train_and_predict(
        self, X_train, y_train, X_val, y_val, X_test, feature_pipeline
    ):
        """
        Main execution method.
        """
        # 1. Fit Feature Pipeline
        feature_pipeline.fit(X_train)

        # 2. Generate OOF (Level 1 Training)
        self.generate_oof_predictions(X_train, y_train, feature_pipeline)

        # 3. Train Meta-Learner (Level 2 Training)
        self.train_meta_learner(y_train)

        # 4. Retrain Final Models (Validation-Guided)
        self.retrain_final_models(X_train, y_train, X_val, y_val, feature_pipeline)

        # 5. Predict Test
        self.predict_test(X_test, feature_pipeline)
