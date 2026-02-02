import os
import numpy as np
import pandas as pd
import joblib
import scipy.sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from library.config import (
    MODEL_DIR,
    N_FOLDS,
    RANDOM_SEED,
    MODEL_TYPES,
    TARGET_COL,
)
from library.model_factory import get_base_models, get_meta_learner


class HybridTrainer:
    """
    Implements the Hybrid Inference Protocol for the Oct-View Stacking Ensemble.
    - Volatile Models (XGB/LGBM): Trained with Early Stopping per fold. All fold models saved.
    - Stable Models (RF/Linear): Trained per fold for OOF, then fully retrained. Single model saved.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.base_models = get_base_models(debug=debug)
        self.meta_learner = get_meta_learner(debug=debug)
        self.n_folds = 2 if debug else N_FOLDS

        # Mapping of model names to their required feature components
        # Keys match those in library.model_factory.get_base_models
        self.feature_map = {
            "lexical_bagger": ["lexical", "meta"],
            "lexical_anchor": ["lexical", "meta"],
            "community_bagger": ["community", "meta"],
            "semantic_booster": ["semantic", "meta"],
            "semantic_gradient": ["semantic", "meta"],
            "semantic_bagger": ["semantic", "meta"],
            "metadata_anchor": ["meta"],
            "temporal_booster": ["meta"],
        }

    def _get_model_features(
        self, model_name, X_lexical, X_community, X_semantic, X_meta
    ):
        """
        Constructs the specific feature matrix for a given model by stacking components.
        """
        components = self.feature_map.get(model_name)
        if not components:
            raise ValueError(f"Unknown model name: {model_name}")

        features_to_stack = []
        is_sparse = False

        for comp in components:
            if comp == "lexical":
                features_to_stack.append(X_lexical)
                is_sparse = True
            elif comp == "community":
                features_to_stack.append(X_community)
                is_sparse = True
            elif comp == "semantic":
                features_to_stack.append(X_semantic)
            elif comp == "meta":
                features_to_stack.append(X_meta)

        if len(features_to_stack) == 1:
            return features_to_stack[0]

        # Stack appropriately
        if is_sparse:
            # If any component is sparse, use scipy.sparse.hstack
            # Ensure all dense components are converted to sparse for stacking if needed,
            # but scipy.sparse.hstack handles dense arrays automatically.
            return scipy.sparse.hstack(features_to_stack).tocsr()
        else:
            return np.hstack(features_to_stack)

    def train(self, df_train, feature_pipeline):
        """
        Main training loop.

        Args:
            df_train: Training DataFrame containing target and raw features.
            feature_pipeline: Fitted FeaturePipeline instance.
        """
        print(f"Starting Hybrid Training (Debug={self.debug})...")

        # 1. Extract Features
        # Note: fit_transform handles caching internally
        X_lexical, X_community, X_semantic, X_meta = feature_pipeline.fit_transform(
            df_train
        )
        y = df_train[TARGET_COL].values

        # 2. Initialize OOF Matrix for Meta-Learner
        # Shape: (n_samples, n_base_models)
        oof_preds = pd.DataFrame(index=df_train.index)

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=RANDOM_SEED
        )

        # 3. Train Base Models
        for model_name, base_model in self.base_models.items():
            print(f"\n--- Training {model_name} ---")

            # Prepare specific features for this model
            X = self._get_model_features(
                model_name, X_lexical, X_community, X_semantic, X_meta
            )

            model_type = MODEL_TYPES.get(model_name, "stable")
            oof_col = np.zeros(len(df_train))

            # Cross-Validation Loop
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                # Clone model for this fold
                model = clone(base_model)

                # Fit Model
                if model_type == "volatile":
                    # Volatile models (XGB, LGBM) use Early Stopping
                    # Note: Parameters like early_stopping_rounds are injected via config/factory
                    # We just need to pass eval_set.
                    model.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_val, y_val)],
                    )

                    # Save Volatile Model (CV-Bagging)
                    model_path = os.path.join(
                        MODEL_DIR, f"{model_name}_fold_{fold}.joblib"
                    )
                    joblib.dump(model, model_path)

                else:
                    # Stable models (RF, Linear) just fit
                    model.fit(X_tr, y_tr)
                    # We do NOT save fold models for stable learners, only the final retrained one.

                # Predict OOF
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X_val)[:, 1]
                else:
                    # Fallback for models that might not have predict_proba (unlikely here)
                    preds = model.predict(X_val)

                oof_col[val_idx] = preds

            # Store OOF predictions
            oof_preds[model_name] = oof_col

            # Calculate and Print Metric
            auc = roc_auc_score(y, oof_col)
            print(f"{model_name} OOF AUC: {auc}")

            # Hybrid Protocol: Retrain Stable Models on Full Data
            if model_type == "stable":
                print(f"Retraining {model_name} on full dataset...")
                final_model = clone(base_model)
                final_model.fit(X, y)

                model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
                joblib.dump(final_model, model_path)
                print(f"Saved stable model to {model_path}")

        # 4. Train Meta-Learner
        print("\n--- Training Meta-Learner ---")
        print(f"OOF Matrix Shape: {oof_preds.shape}")

        self.meta_learner.fit(oof_preds, y)

        meta_auc = roc_auc_score(y, self.meta_learner.predict_proba(oof_preds)[:, 1])
        print(f"Meta-Learner CV AUC (on OOF): {meta_auc}")

        # Save Meta-Learner
        meta_path = os.path.join(MODEL_DIR, "meta_learner.joblib")
        joblib.dump(self.meta_learner, meta_path)
        print(f"Saved Meta-Learner to {meta_path}")

        print("\nTraining Complete.")
