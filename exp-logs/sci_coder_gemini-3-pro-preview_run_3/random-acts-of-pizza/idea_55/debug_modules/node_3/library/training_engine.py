import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import logging
import joblib

from library.config import Config
from library.utils import save_model, load_model, set_seed
from library.model_factory import (
    get_base_learner,
    get_meta_learner,
    is_volatile,
    prepare_model_inputs,
)


class HybridTrainer:
    """
    Implements the Hybrid Inference Protocol for training a Stacking Ensemble.
    Handles Volatile (CV-Bagging) and Stable (Full-Retrain) strategies differently.
    """

    def __init__(self):
        self.oof_predictions = pd.DataFrame()
        self.y_full = None
        self.base_models = Config.STABLE_MODELS + Config.VOLATILE_MODELS

    def _merge_data(self, train_feats, train_y, val_feats, val_y):
        """
        Concatenates train and validation sets into a full dataset for CV.
        """
        merged_feats = {}
        # Keys are expected to be 'X_lexical', 'X_behavioral', 'X_semantic', 'X_metadata'
        keys = train_feats.keys()

        for key in keys:
            t_data = train_feats[key]
            v_data = val_feats[key]
            merged_feats[key] = np.vstack([t_data, v_data])

        merged_y = np.concatenate([train_y, val_y])

        return merged_feats, merged_y

    def _get_early_stopping_params(self, model_name):
        """
        Retrieves early stopping parameters from Config if they exist.
        """
        param_name = f"{model_name.upper()}_PARAMS"
        if hasattr(Config, param_name):
            params = getattr(Config, param_name)
            if "early_stopping_rounds" in params:
                return params["early_stopping_rounds"]
        return None

    def train_stacking_layer(self, train_feats, train_y, val_feats, val_y):
        """
        Trains Level 1 Base Learners.

        Strategy:
        - Volatile: 5-Fold CV with Early Stopping. Save ALL 5 fold models.
        - Stable: 5-Fold CV for OOF only. Retrain SINGLE model on full data.
        """
        logging.info("Starting Level 1 Stacking Layer Training...")
        set_seed()

        # 1. Merge Train and Val for Unified Cross-Validation
        X_full_dict, y_full = self._merge_data(train_feats, train_y, val_feats, val_y)
        self.y_full = y_full

        # Initialize OOF DataFrame
        n_samples = y_full.shape[0]
        self.oof_predictions = pd.DataFrame(index=range(n_samples))

        # Setup CV
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE
        )

        for model_name in self.base_models:
            logging.info(f"Training Base Learner: {model_name}")

            # Prepare specific feature subset for this model
            X_model_full = prepare_model_inputs(X_full_dict, model_name)

            # Storage for OOF preds for this model
            model_oof = np.zeros(n_samples)

            is_vol = is_volatile(model_name)
            es_rounds = self._get_early_stopping_params(model_name)

            # Cross-Validation Loop
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(
                skf.split(X_model_full, y_full)
            ):
                X_train_fold, X_val_fold = (
                    X_model_full[train_idx],
                    X_model_full[val_idx],
                )
                y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]

                learner = get_base_learner(model_name)

                # Fit Logic
                if is_vol and es_rounds:
                    # Volatile: Use Early Stopping
                    fit_kwargs = {"eval_set": [(X_val_fold, y_val_fold)]}
                    model_type = str(type(learner)).lower()

                    if "xgb" in model_type:
                        # XGBoost: Move params to set_params (Cite debug_lesson_10)
                        learner.set_params(
                            eval_metric="auc", early_stopping_rounds=es_rounds
                        )
                        fit_kwargs["verbose"] = False
                    elif "lgbm" in model_type:
                        # LightGBM: Use callbacks (Cite debug_lesson_17)
                        import lightgbm as lgb

                        callbacks = [
                            lgb.early_stopping(
                                stopping_rounds=es_rounds, verbose=False
                            ),
                            lgb.log_evaluation(period=0),
                        ]
                        fit_kwargs["callbacks"] = callbacks
                        fit_kwargs["eval_metric"] = "auc"
                    else:
                        fit_kwargs["early_stopping_rounds"] = es_rounds
                        fit_kwargs["eval_metric"] = "auc"

                    learner.fit(X_train_fold, y_train_fold, **fit_kwargs)
                else:
                    # Stable: Standard Fit
                    learner.fit(X_train_fold, y_train_fold)

                # Predict OOF
                if hasattr(learner, "predict_proba"):
                    preds = learner.predict_proba(X_val_fold)[:, 1]
                else:
                    preds = learner.predict(X_val_fold)

                model_oof[val_idx] = preds

                # Score
                score = roc_auc_score(y_val_fold, preds)
                fold_scores.append(score)
                logging.info(f"  {model_name} [Fold {fold}] AUC: {score}")

                # Save Volatile Models (Per Fold)
                if is_vol:
                    save_model(learner, model_name, is_volatile=True, fold=fold)

            avg_score = np.mean(fold_scores)
            logging.info(f"  {model_name} Average CV AUC: {avg_score}")

            # Store OOF
            self.oof_predictions[model_name] = model_oof

            # Stable Models: Retrain on Full Data
            if not is_vol:
                logging.info(
                    f"  {model_name}: Retraining on Full Dataset (Stable Strategy)..."
                )
                full_learner = get_base_learner(model_name)
                full_learner.fit(X_model_full, y_full)
                save_model(full_learner, model_name, is_volatile=False)

        logging.info("Level 1 Training Complete.")

    def train_meta_learner(self):
        """
        Trains the Level 2 Meta-Learner on OOF predictions.
        """
        logging.info("Training Level 2 Meta-Learner...")

        if self.oof_predictions.empty or self.y_full is None:
            raise ValueError(
                "OOF predictions not found. Run train_stacking_layer first."
            )

        X_meta = self.oof_predictions.values
        y_meta = self.y_full

        meta_learner = get_meta_learner()
        meta_learner.fit(X_meta, y_meta)

        # Evaluate In-Sample (Just for sanity check)
        preds = meta_learner.predict_proba(X_meta)[:, 1]
        score = roc_auc_score(y_meta, preds)
        logging.info(f"Meta-Learner OOF CV AUC Score: {score}")

        save_model(meta_learner, "meta_learner")
        logging.info("Meta-Learner Saved.")

    def generate_predictions(self, test_feats, ids=None):
        """
        Generates predictions for the test set using the Hybrid Inference Protocol.

        - Volatile: Load 5 fold models, predict, average.
        - Stable: Load 1 full model, predict.
        - Stack: Feed to Meta-Learner.
        """
        logging.info("Generating Final Predictions...")
        set_seed()

        n_samples = test_feats["X_metadata"].shape[0]
        level1_preds = pd.DataFrame(index=range(n_samples))

        for model_name in self.base_models:
            logging.info(f"Inference: {model_name}")

            # Prepare Input
            X_test_model = prepare_model_inputs(test_feats, model_name)

            is_vol = is_volatile(model_name)

            if is_vol:
                # Volatile: CV-Bagging (Average of 5 folds)
                fold_preds = np.zeros(n_samples)
                for fold in range(Config.N_FOLDS):
                    learner = load_model(model_name, is_volatile=True, fold=fold)
                    if hasattr(learner, "predict_proba"):
                        p = learner.predict_proba(X_test_model)[:, 1]
                    else:
                        p = learner.predict(X_test_model)
                    fold_preds += p

                level1_preds[model_name] = fold_preds / Config.N_FOLDS

            else:
                # Stable: Single Full Model
                learner = load_model(model_name, is_volatile=False)
                if hasattr(learner, "predict_proba"):
                    p = learner.predict_proba(X_test_model)[:, 1]
                else:
                    p = learner.predict(X_test_model)
                level1_preds[model_name] = p

        # Level 2 Inference
        logging.info("Inference: Meta-Learner")
        meta_learner = load_model("meta_learner")
        X_meta_test = level1_preds.values
        final_probs = meta_learner.predict_proba(X_meta_test)[:, 1]

        # Create Submission DataFrame
        # Note: We need request_ids. Assuming they are not passed here but handled externally
        # or we just return the probabilities.
        # Requirement says: "Save the final predictions to ./submission/submission.csv"
        # But we need the IDs. The test_feats dict usually contains arrays, not IDs.
        # We will assume the caller handles ID mapping or we load the test parquet to get IDs.

        # Load test parquet to get IDs if not provided
        if ids is None:
            test_df = pd.read_parquet(Config.TEST_PATH)
            ids = test_df[Config.ID_COL]

        # Ensure ids align with predictions (ignore index if Series)
        if hasattr(ids, "values"):
            ids = ids.values

        submission = pd.DataFrame({Config.ID_COL: ids, Config.TARGET_COL: final_probs})

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        logging.info(f"Submission saved to {Config.SUBMISSION_PATH}")

        return submission
