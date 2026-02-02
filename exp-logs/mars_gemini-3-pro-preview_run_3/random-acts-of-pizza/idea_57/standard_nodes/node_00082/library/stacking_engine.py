import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from library.config import Config
from library.utils import (
    setup_logger,
    save_model,
    load_model,
    compute_metric,
    print_metric,
)
from library.model_zoo import get_model

# Initialize Logger
logger = setup_logger("stacking_engine")


class HybridEnsembleTrainer:
    """
    Orchestrates the training and inference of the Hybrid Stacking Ensemble.

    Implements:
    1. Feature Concatenation (Base Features + Metadata).
    2. 5-Fold Stratified CV for OOF Generation.
    3. Hybrid Training Strategy:
       - Volatile Learners (Boosters): Save per-fold models.
       - Stable Learners (RF/Linear): Save per-fold (for consistency) AND retrain on full data.
    4. Meta-Learner Training.
    5. Hybrid Inference Strategy:
       - Volatile: CV-Bagging (Average of 5 folds).
       - Stable: Prediction from single retrained model.
    """

    def __init__(self, feature_data, union_train_df, test_df):
        """
        Args:
            feature_data (dict): Dictionary containing tuples of (train, test) features.
            union_train_df (pd.DataFrame): The merged training and validation dataframe.
            test_df (pd.DataFrame): The test dataframe.
        """
        self.feature_data = feature_data
        self.union_train_df = union_train_df
        self.test_df = test_df

        # Targets and IDs
        self.y = union_train_df[Config.TARGET_COL].values
        self.train_ids = union_train_df[Config.ID_COL].values
        self.test_ids = test_df[Config.ID_COL].values

        # Model categorization
        self.volatile_types = ["xgboost", "lightgbm"]
        self.stable_types = ["sklearn_rf", "sklearn_lr"]

        # Placeholders for OOF and Meta-model
        self.oof_df = pd.DataFrame({Config.ID_COL: self.train_ids})
        self.meta_learner = None

    def _prepare_X(self, model_key, indices=None, is_test=False):
        """
        Prepares the feature matrix X for a specific model.
        Concatenates the specific modality features with the dense metadata vector.

        Args:
            model_key (str): Key from Config.MODEL_CONFIGS.
            indices (np.array, optional): Indices to slice the training data.
            is_test (bool): Whether to retrieve test data.

        Returns:
            X (sparse matrix or numpy array): The feature matrix.
        """
        model_conf = Config.MODEL_CONFIGS[model_key]
        feature_set_name = model_conf["feature_set"]

        # Select Train (0) or Test (1) index from the tuple stored in feature_data
        tuple_idx = 1 if is_test else 0

        # Base Features
        X_base = self.feature_data[feature_set_name][tuple_idx]

        # Metadata Features (Always concatenate unless the base IS metadata)
        X_meta = self.feature_data["metadata_only"][tuple_idx]

        # Slice if indices provided (only for training data)
        if indices is not None and not is_test:
            if sp.issparse(X_base):
                X_base = X_base[indices]
            else:
                X_base = X_base[indices]
            X_meta = X_meta[indices]

        # Concatenation Logic
        if feature_set_name == "metadata_only":
            # Redundant to concatenate metadata to itself
            return X_meta

        if sp.issparse(X_base):
            # Sparse + Dense -> Sparse
            X_combined = sp.hstack([X_base, sp.csr_matrix(X_meta)], format="csr")
        else:
            # Dense + Dense -> Dense
            X_combined = np.hstack([X_base, X_meta])

        return X_combined

    def _calculate_scale_pos_weight(self, y_train):
        """Calculates scale_pos_weight for XGBoost based on class imbalance."""
        n_pos = np.sum(y_train)
        n_neg = len(y_train) - n_pos
        return n_neg / n_pos if n_pos > 0 else 1.0

    def train_level_1(self):
        """
        Performs 5-Fold Stratified CV to train Level 1 Base Learners.
        Generates OOF predictions and saves models for each fold.
        """
        logger.info("Starting Level 1 Training (5-Fold CV)...")

        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE
        )

        # Initialize OOF columns
        for model_key in Config.MODEL_CONFIGS.keys():
            self.oof_df[model_key] = 0.0

        fold = 0
        for train_idx, val_idx in skf.split(self.union_train_df, self.y):
            logger.info(f"--- Processing Fold {fold} ---")

            y_train_fold = self.y[train_idx]
            y_val_fold = self.y[val_idx]

            for model_key, config in Config.MODEL_CONFIGS.items():
                logger.info(f"Training {model_key} (Fold {fold})...")

                # Prepare Data
                X_train = self._prepare_X(model_key, indices=train_idx)
                X_val = self._prepare_X(model_key, indices=val_idx)

                # Dynamic Hyperparameters
                kwargs = {}
                if config["type"] == "xgboost":
                    spw = self._calculate_scale_pos_weight(y_train_fold)
                    kwargs["scale_pos_weight"] = spw

                # Instantiate Model
                model = get_model(model_key, **kwargs)

                # Fit Model
                if config["type"] in self.volatile_types:
                    # Volatile: Use Early Stopping
                    fit_params = {
                        "eval_set": [(X_val, y_val_fold)],
                    }

                    # LightGBM 4.0+ removed verbose from fit; XGBoost still accepts it
                    if config["type"] != "lightgbm":
                        fit_params["verbose"] = False

                    model.fit(X_train, y_train_fold, **fit_params)
                else:
                    # Stable: Standard Fit
                    model.fit(X_train, y_train_fold)

                # Predict OOF (Probabilities for class 1)
                oof_preds = model.predict_proba(X_val)[:, 1]

                # Store OOF
                # We map val_idx back to the dataframe index
                # self.oof_df is initialized with train_ids order, which matches self.y order
                self.oof_df.loc[val_idx, model_key] = oof_preds

                # Save Model (Persistence for all folds)
                save_model(model, model_key, fold=fold)

                # Metric for this fold/model
                auc = compute_metric(y_val_fold, oof_preds)
                print_metric(f"{model_key} Fold {fold} AUC", auc)

            fold += 1

        logger.info("Level 1 Training Complete.")

        # Save OOF DataFrame for analysis
        oof_path = os.path.join(Config.WORKING_DIR, "oof_predictions.csv")
        self.oof_df.to_csv(oof_path, index=False)

    def retrain_stable_models(self):
        """
        Retrains 'Stable' learners (RF, LR) on the full Union Dataset.
        This maximizes data usage for models that don't require early stopping validation sets.
        """
        logger.info("Retraining Stable Models on Full Union Dataset...")

        for model_key, config in Config.MODEL_CONFIGS.items():
            if config["type"] in self.stable_types:
                logger.info(f"Retraining {model_key} (Full Data)...")

                # Prepare Full Data
                X_full = self._prepare_X(
                    model_key, indices=None
                )  # indices=None gets full array

                # Instantiate
                model = get_model(model_key)

                # Fit
                model.fit(X_full, self.y)

                # Save as the 'main' model (no fold suffix)
                save_model(model, model_key, fold=None)

    def train_meta_learner(self):
        """
        Trains the Level 2 Logistic Regression Meta-Learner on the OOF predictions.
        """
        logger.info("Training Level 2 Meta-Learner...")

        # Construct Feature Matrix for Meta Learner
        # Order of columns must be consistent
        feature_cols = list(Config.MODEL_CONFIGS.keys())
        X_meta_train = self.oof_df[feature_cols].values
        y_meta_train = self.y

        # Instantiate and Fit
        self.meta_learner = get_model("meta_learner")
        self.meta_learner.fit(X_meta_train, y_meta_train)

        # Evaluate on OOF (In-sample for meta learner, but OOF for base learners)
        preds = self.meta_learner.predict_proba(X_meta_train)[:, 1]
        auc = compute_metric(y_meta_train, preds)
        print_metric("Meta-Learner CV AUC", auc)

        # Save Meta Learner
        save_model(self.meta_learner, "meta_learner")

    def predict(self):
        """
        Generates final predictions for the Test set using the Hybrid Inference Strategy.

        1. Volatile Models: Average predictions from all 5 saved fold-models (CV-Bagging).
        2. Stable Models: Use predictions from the single fully-retrained model.
        3. Meta Learner: Combine base predictions.

        Returns:
            np.array: Final probabilities for the test set.
        """
        logger.info("Generating Final Predictions...")

        test_preds_dict = {}

        for model_key, config in Config.MODEL_CONFIGS.items():
            logger.info(f"Predicting with {model_key}...")

            # Prepare Test Data
            X_test = self._prepare_X(model_key, is_test=True)

            if config["type"] in self.volatile_types:
                # Volatile: CV-Bagging
                fold_preds = []
                for fold in range(Config.N_FOLDS):
                    model = load_model(model_key, fold=fold)
                    p = model.predict_proba(X_test)[:, 1]
                    fold_preds.append(p)

                # Average
                avg_preds = np.mean(fold_preds, axis=0)
                test_preds_dict[model_key] = avg_preds

            else:
                # Stable: Use Retrained Model
                # Load the model saved without fold suffix
                model = load_model(model_key, fold=None)
                p = model.predict_proba(X_test)[:, 1]
                test_preds_dict[model_key] = p

        # Construct Meta Matrix
        feature_cols = list(Config.MODEL_CONFIGS.keys())
        X_meta_test = pd.DataFrame(test_preds_dict)[feature_cols].values

        # Final Prediction
        final_probs = self.meta_learner.predict_proba(X_meta_test)[:, 1]

        return final_probs
