import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import get_logger, timer
from library.feature_engineering import FeaturePipeline
from library.model_definitions import get_base_learners, get_meta_learner


class CVEnsembleTrainer:
    """
    Orchestrates the CV-Bagging training protocol for the Hex-View Stacking Ensemble.
    Trains base learners on 5 folds, persists all models, trains a meta-learner,
    and generates the final submission.
    """

    def __init__(self):
        self.logger = get_logger("CVEnsembleTrainer")
        self.output_dir = Config.WORKING_DIR
        self.models_dir = os.path.join(self.output_dir, "models")
        self.predictions_dir = os.path.join(self.output_dir, "predictions")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)

        # Load Data
        self.pipeline = FeaturePipeline()
        self.X_train_dict, self.y_train, self.X_test_dict, self.test_ids = (
            self.pipeline.get_data(load_cached_data=True)
        )

        # Define Model Registry
        self.base_learners_factories = get_base_learners()
        self.model_names = list(self.base_learners_factories.keys())

    def _get_model_features(self, X_dict, model_name, indices=None):
        """
        Retrieves and concatenates the specific feature views required for a given model.

        Args:
            X_dict (dict): Dictionary of feature matrices (train or test).
            model_name (str): Name of the model to determine feature composition.
            indices (np.array, optional): Indices to slice the data (for CV splits).

        Returns:
            np.ndarray or scipy.sparse.csr_matrix: The combined feature matrix.
        """
        # 1. Lexical Bagger: Sparse Lexical + Dense Contextual
        if model_name == "lexical_bagger":
            lex = X_dict["lexical"]
            ctx = X_dict["contextual"]
            if indices is not None:
                lex = lex[indices]
                ctx = ctx[indices]
            # Stack sparse and dense (converted to sparse)
            return sp.hstack([lex, sp.csr_matrix(ctx)])

        # 2. Community Bagger: Sparse Behavioral + Dense Contextual
        elif model_name == "community_bagger":
            beh = X_dict["behavioral"]
            ctx = X_dict["contextual"]
            if indices is not None:
                beh = beh[indices]
                ctx = ctx[indices]
            return sp.hstack([beh, sp.csr_matrix(ctx)])

        # 3. Semantic Models: Dense Semantic + Dense Contextual
        elif model_name in ["semantic_booster", "semantic_bagger"]:
            sem = X_dict["semantic"]
            ctx = X_dict["contextual"]
            if indices is not None:
                sem = sem[indices]
                ctx = ctx[indices]
            return np.hstack([sem, ctx])

        # 4. Contextual Models: Dense Contextual Only
        elif model_name in ["metadata_anchor", "temporal_booster"]:
            ctx = X_dict["contextual"]
            if indices is not None:
                ctx = ctx[indices]
            return ctx

        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def train_loop(self):
        """
        Executes the 5-Fold CV-Bagging training loop.
        """
        with timer("CV-Bagging Training Loop"):
            skf = StratifiedKFold(
                n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
            )

            # Initialize OOF container
            # Shape: (n_samples, n_base_learners)
            oof_preds = pd.DataFrame(
                np.zeros((len(self.y_train), len(self.model_names))),
                columns=self.model_names,
            )

            # Iterate Folds
            for fold, (train_idx, val_idx) in enumerate(
                skf.split(np.zeros(len(self.y_train)), self.y_train)
            ):
                self.logger.info(f"--- Starting Fold {fold + 1}/{Config.N_FOLDS} ---")

                y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]

                # Train each base learner
                for model_name in self.model_names:
                    # Prepare Features
                    X_tr = self._get_model_features(
                        self.X_train_dict, model_name, train_idx
                    )
                    X_val = self._get_model_features(
                        self.X_train_dict, model_name, val_idx
                    )

                    # Instantiate Model
                    model_factory = self.base_learners_factories[model_name]
                    model = model_factory()

                    # Train
                    # Handle Early Stopping for Boosters
                    if model_name == "semantic_booster":  # XGBoost
                        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                    elif model_name == "temporal_booster":  # LightGBM
                        # LightGBM requires callbacks for early stopping in recent versions or specific params
                        # We use the params defined in Config which include early_stopping_rounds
                        # but we need to pass eval_set here.
                        # Note: sklearn API for LGBM usually accepts eval_set in fit
                        model.fit(
                            X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc"
                        )
                    else:
                        # RF / LogReg
                        model.fit(X_tr, y_tr)

                    # Predict (OOF)
                    # Handle probability prediction
                    if hasattr(model, "predict_proba"):
                        p_val = model.predict_proba(X_val)[:, 1]
                    else:
                        # Fallback if needed, though all classifiers should have it
                        p_val = model.predict(X_val)

                    # Store OOF
                    oof_preds.loc[val_idx, model_name] = p_val

                    # Save Model (Persistence for Bagging)
                    model_path = os.path.join(
                        self.models_dir, f"{model_name}_fold_{fold}.joblib"
                    )
                    joblib.dump(model, model_path)

                    # Log Fold Score
                    fold_auc = roc_auc_score(y_val, p_val)
                    # self.logger.info(f"Fold {fold+1} - {model_name} AUC: {fold_auc:.6f}")

            # Save OOF predictions
            oof_path = os.path.join(self.predictions_dir, "oof_predictions.parquet")
            oof_preds.to_parquet(oof_path)

            # Print Summary Metrics
            self.logger.info("\n--- OOF Performance Summary ---")
            for model_name in self.model_names:
                auc = roc_auc_score(self.y_train, oof_preds[model_name])
                self.logger.info(f"{model_name}: OOF AUC = {auc:.10f}")

    def train_meta_learner(self):
        """
        Trains the Level 2 Meta-Learner on OOF predictions.
        """
        with timer("Meta-Learner Training"):
            # Load OOF
            oof_path = os.path.join(self.predictions_dir, "oof_predictions.parquet")
            if not os.path.exists(oof_path):
                raise FileNotFoundError(
                    "OOF predictions not found. Run train_loop first."
                )

            oof_preds = pd.read_parquet(oof_path)
            X_meta = oof_preds[self.model_names].values
            y = self.y_train

            # Train Meta Learner
            meta_model = get_meta_learner()
            meta_model.fit(X_meta, y)

            # Evaluate on OOF (Proxy for performance)
            meta_preds = meta_model.predict_proba(X_meta)[:, 1]
            auc = roc_auc_score(y, meta_preds)
            self.logger.info(f"Meta-Learner OOF AUC: {auc:.10f}")

            # Show Weights
            weights = dict(zip(self.model_names, meta_model.coef_[0]))
            self.logger.info(f"Meta-Learner Weights: {weights}")

            # Save Meta Model
            joblib.dump(
                meta_model, os.path.join(self.models_dir, "meta_learner.joblib")
            )

    def generate_submission(self):
        """
        Generates predictions for the test set using CV-Bagging and the Meta-Learner.
        """
        with timer("Submission Generation"):
            # 1. Generate Bagged Predictions for each Base Learner
            test_meta_features = pd.DataFrame(
                index=range(len(self.test_ids)), columns=self.model_names
            )

            for model_name in self.model_names:
                # Prepare Test Features for this model type
                X_test = self._get_model_features(self.X_test_dict, model_name)

                fold_preds = []
                for fold in range(Config.N_FOLDS):
                    model_path = os.path.join(
                        self.models_dir, f"{model_name}_fold_{fold}.joblib"
                    )
                    model = joblib.load(model_path)

                    if hasattr(model, "predict_proba"):
                        p_test = model.predict_proba(X_test)[:, 1]
                    else:
                        p_test = model.predict(X_test)

                    fold_preds.append(p_test)

                # Average across folds (Bagging)
                avg_pred = np.mean(fold_preds, axis=0)
                test_meta_features[model_name] = avg_pred

            # 2. Meta-Learner Prediction
            meta_model_path = os.path.join(self.models_dir, "meta_learner.joblib")
            meta_model = joblib.load(meta_model_path)

            X_meta_test = test_meta_features[self.model_names].values
            final_preds = meta_model.predict_proba(X_meta_test)[:, 1]

            # 3. Create Submission File
            submission = pd.DataFrame(
                {"request_id": self.test_ids, "requester_received_pizza": final_preds}
            )

            submission.to_csv(Config.SUBMISSION_PATH, index=False)
            self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
            self.logger.info(f"Submission shape: {submission.shape}")
            self.logger.info(f"Head:\n{submission.head()}")

    def run(self):
        """
        Execute the full pipeline.
        """
        self.train_loop()
        self.train_meta_learner()
        self.generate_submission()
