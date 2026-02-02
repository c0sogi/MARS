import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library import config
from library import utils
from library.models import ModelFactory


class EnsembleEngine:
    """
    Orchestrates the Hept-View Full-Spectrum Stacking Ensemble.
    Implements the Hybrid Inference Protocol:
    - Stable models: Retrained on full data.
    - Volatile models: Bagged across CV folds.
    """

    def __init__(self, data: dict):
        self.data = data
        self.models_dir = os.path.join(config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        # Containers for Level 1 predictions
        self.oof_matrix = pd.DataFrame(index=range(len(data["y_train"])))
        self.test_matrix = pd.DataFrame(index=range(len(data["test_ids"])))

        # Track which models have been trained
        self.trained_models_registry = []

    def _get_fit_params(self, model_name: str, X_val, y_val) -> dict:
        """
        Constructs fit parameters, specifically handling early stopping
        for volatile gradient boosting models.
        """
        fit_params = {}
        raw_params = config.MODEL_PARAMS.get(model_name, {})
        es_rounds = raw_params.get("early_stopping_rounds")

        if es_rounds:
            # Both XGBoost and LightGBM sklearn APIs support eval_set and early_stopping_rounds in fit()
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["early_stopping_rounds"] = es_rounds

            model_type = ModelFactory.get_model_config(model_name)["type"]

            if model_type == "xgb":
                fit_params["verbose"] = False
            elif model_type == "lgbm":
                fit_params["eval_metric"] = "auc"
                # LightGBM might warn about callbacks, but direct kwarg is robust for this version

        return fit_params

    def run_cv_and_training(self):
        """
        Executes the training pipeline.
        1. Runs 5-Fold CV for all models to get OOF predictions.
        2. Persists models according to their volatility (Fold-based vs Full-retrain).
        """
        utils.set_seed()
        y = self.data["y_train"]
        skf = StratifiedKFold(
            n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
        )

        print(f"Starting Hept-View Ensemble Training on {len(y)} samples...")

        for model_name in ModelFactory.MODEL_REGISTRY.keys():
            model_conf = ModelFactory.get_model_config(model_name)
            volatility = model_conf["volatility"]
            print(f"\nTraining {model_name} [{volatility}]...")

            # Prepare Features for this specific model branch
            X = ModelFactory.prepare_features(self.data, model_name, split="train")

            oof_preds = np.zeros(len(y))
            fold_aucs = []

            # --- Cross Validation Loop ---
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                # Instantiate fresh model
                model = ModelFactory.get_model(model_name)

                # Get specific fit params (e.g. early stopping)
                fit_params = {}
                if volatility == "volatile":
                    fit_params = self._get_fit_params(model_name, X_val, y_val)

                model.fit(X_tr, y_tr, **fit_params)

                # Predict
                # Handle probability prediction safely
                if hasattr(model, "predict_proba"):
                    val_probs = model.predict_proba(X_val)[:, 1]
                else:
                    # Fallback for models that might not have predict_proba (unlikely here)
                    val_probs = model.predict(X_val)

                oof_preds[val_idx] = val_probs

                # Metric
                auc = roc_auc_score(y_val, val_probs)
                fold_aucs.append(auc)

                # Hybrid Persistence Strategy:
                # If Volatile, we MUST save the fold model for inference averaging
                if volatility == "volatile":
                    model_path = os.path.join(
                        self.models_dir, f"{model_name}_fold_{fold}.joblib"
                    )
                    joblib.dump(model, model_path)

            # Report CV Results
            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            print(f"  CV AUC: {mean_auc:.10f} (Std: {std_auc:.4f})")

            # Store OOF predictions
            self.oof_matrix[model_name] = oof_preds

            # Hybrid Retraining Strategy:
            # If Stable, we discard fold models and retrain on FULL data for maximum signal
            if volatility == "stable":
                print(f"  Retraining {model_name} on full dataset...")
                full_model = ModelFactory.get_model(model_name)
                full_model.fit(X, y)
                model_path = os.path.join(self.models_dir, f"{model_name}_full.joblib")
                joblib.dump(full_model, model_path)

            self.trained_models_registry.append(model_name)

    def train_meta_learner(self):
        """
        Trains the Level 2 Logistic Regression on the OOF predictions from Level 1.
        """
        print("\nTraining Meta-Learner...")
        X_meta = self.oof_matrix.values
        y = self.data["y_train"]

        meta_model = ModelFactory.get_meta_learner()
        meta_model.fit(X_meta, y)

        # Print coefficients to see which L1 models are driving the ensemble
        print("  Meta-Learner Coefficients:")
        for name, coef in zip(self.oof_matrix.columns, meta_model.coef_[0]):
            print(f"    {name}: {coef:.6f}")

        joblib.dump(meta_model, os.path.join(self.models_dir, "meta_learner.joblib"))

    def generate_submission(self):
        """
        Generates predictions for the test set.
        Applies the Hybrid Inference Protocol:
        - Stable: Single model prediction.
        - Volatile: Average of 5 fold model predictions.
        """
        print("\nGenerating Test Predictions...")

        # 1. Generate Level 1 Test Predictions
        for model_name in self.trained_models_registry:
            model_conf = ModelFactory.get_model_config(model_name)
            volatility = model_conf["volatility"]

            # Prepare Test Features
            X_test = ModelFactory.prepare_features(self.data, model_name, split="test")

            if volatility == "stable":
                # Load single full model
                model_path = os.path.join(self.models_dir, f"{model_name}_full.joblib")
                model = joblib.load(model_path)
                preds = model.predict_proba(X_test)[:, 1]
            else:
                # Load all fold models and average (CV-Bagging)
                fold_preds = []
                for fold in range(config.N_FOLDS):
                    model_path = os.path.join(
                        self.models_dir, f"{model_name}_fold_{fold}.joblib"
                    )
                    model = joblib.load(model_path)
                    fold_preds.append(model.predict_proba(X_test)[:, 1])
                preds = np.mean(fold_preds, axis=0)

            self.test_matrix[model_name] = preds

        # 2. Generate Level 2 Final Predictions
        meta_model = joblib.load(os.path.join(self.models_dir, "meta_learner.joblib"))
        final_probs = meta_model.predict_proba(self.test_matrix.values)[:, 1]

        # 3. Save Submission
        submission = pd.DataFrame(
            {
                "request_id": self.data["test_ids"],
                "requester_received_pizza": final_probs,
            }
        )

        print(f"Saving submission to {config.SUBMISSION_PATH}...")
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print("Submission generated successfully.")
