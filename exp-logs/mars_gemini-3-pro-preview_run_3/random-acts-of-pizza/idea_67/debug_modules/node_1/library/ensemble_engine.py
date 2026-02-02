import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.feature_extraction import FeaturePipeline
from library.model_registry import ModelRegistry


class DecaViewEnsemble:
    def __init__(self):
        self.pipeline = FeaturePipeline()
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        self.submission_dir = Config.SUBMISSION_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # Define model list based on architecture
        self.model_names = [
            # Branch 1: Sparse Lexical
            "lexical_bagger",
            "lexical_randomizer",
            "lexical_anchor",
            # Branch 2: Sparse Behavioral
            "community_bagger",
            "community_anchor",
            # Branch 3: Dense Semantic
            "semantic_booster",
            "semantic_gradient",
            "semantic_bagger",
            # Branch 4: Contextual
            "metadata_anchor",
            "temporal_booster",
        ]

    def _get_features_for_model(self, model_name, feature_dict, indices=None):
        """
        Routes the correct feature subset to the model based on its branch.
        Optionally slices the data by indices (for CV folds).
        """
        # Determine input type
        if "lexical" in model_name:
            X = self.pipeline.get_lexical_input(feature_dict)
        elif "community" in model_name:
            X = self.pipeline.get_community_input(feature_dict)
        elif "semantic" in model_name:
            X = self.pipeline.get_semantic_input(feature_dict)
        elif "metadata" in model_name or "temporal" in model_name:
            X = self.pipeline.get_metadata_input(feature_dict)
        else:
            raise ValueError(f"Unknown feature mapping for model: {model_name}")

        # Slice if indices provided
        if indices is not None:
            return X[indices]
        return X

    def run(self, df_train, df_test):
        """
        Executes the full pipeline: Feature Extraction -> CV Training -> Retraining -> Inference.
        """
        print("Starting DecaView Ensemble Pipeline...")

        # 1. Feature Extraction
        print("--- Step 1: Feature Extraction ---")
        features = self.pipeline.execute(df_train, df_test, load_cached_data=True)
        train_feats = features["train"]
        test_feats = features["test"]

        y = df_train[Config.TARGET_COL].values

        # 2. Cross-Validation Loop (Level 1 Training & OOF Generation)
        print("--- Step 2: Level 1 Cross-Validation ---")
        oof_preds = pd.DataFrame(index=df_train.index, columns=self.model_names)
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            print(f"\nProcessing Fold {fold + 1}/{Config.N_FOLDS}")

            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            for model_name in self.model_names:
                model_type = ModelRegistry.get_model_type(model_name)
                model = ModelRegistry.create_model(model_name)

                # Get specific features for this fold
                X_train_fold = self._get_features_for_model(
                    model_name, train_feats, train_idx
                )
                X_val_fold = self._get_features_for_model(
                    model_name, train_feats, val_idx
                )

                # Train
                if model_type == "volatile":
                    # Volatile models (XGB/LGBM) use Early Stopping
                    eval_set = [(X_val_fold, y_val_fold)]

                    # Handle different library signatures if necessary, but fit usually works
                    # XGBoost and LightGBM scikit-learn API supports eval_set and early_stopping_rounds
                    # Note: early_stopping_rounds is in params for XGB/LGBM in Config
                    model.fit(
                        X_train_fold, y_train_fold, eval_set=eval_set, eval_metric="auc"
                    )
                else:
                    # Stable models
                    model.fit(X_train_fold, y_train_fold)

                # Predict OOF
                if hasattr(model, "predict_proba"):
                    p_val = model.predict_proba(X_val_fold)[:, 1]
                else:
                    p_val = model.predict(X_val_fold)

                oof_preds.loc[val_idx, model_name] = p_val

                # Save Fold Model
                # We save ALL fold models.
                # Volatile needs them for inference (CV-Bagging).
                # Stable needs them just for OOF consistency, but we retrain stable later.
                # We save them anyway for safety.
                joblib.dump(
                    model,
                    os.path.join(self.models_dir, f"{model_name}_fold_{fold}.joblib"),
                )

        # Evaluate OOF Performance
        print("\n--- OOF Performance ---")
        for model_name in self.model_names:
            score = roc_auc_score(y, oof_preds[model_name])
            print(f"{model_name}: ROC AUC = {score}")

        # 3. Meta-Learner Training (Level 2)
        print("\n--- Step 3: Meta-Learner Training ---")
        meta_learner = ModelRegistry.get_meta_learner()
        meta_learner.fit(oof_preds.values, y)

        meta_score = roc_auc_score(
            y, meta_learner.predict_proba(oof_preds.values)[:, 1]
        )
        print(f"Meta-Learner CV Score: {meta_score}")

        joblib.dump(meta_learner, os.path.join(self.models_dir, "meta_learner.joblib"))

        # 4. Retrain Stable Models on Full Data
        print("\n--- Step 4: Retraining Stable Models on Full Data ---")
        for model_name in self.model_names:
            if ModelRegistry.get_model_type(model_name) == "stable":
                print(f"Retraining {model_name}...")
                model = ModelRegistry.create_model(model_name)
                X_full = self._get_features_for_model(model_name, train_feats)
                model.fit(X_full, y)
                joblib.dump(
                    model, os.path.join(self.models_dir, f"{model_name}.joblib")
                )

        # 5. Inference on Test Set
        print("\n--- Step 5: Inference ---")
        test_level1 = pd.DataFrame(index=df_test.index, columns=self.model_names)

        for model_name in self.model_names:
            model_type = ModelRegistry.get_model_type(model_name)
            X_test = self._get_features_for_model(model_name, test_feats)

            if model_type == "volatile":
                # CV-Bagging: Load all 5 fold models and average
                fold_preds = []
                for fold in range(Config.N_FOLDS):
                    model_path = os.path.join(
                        self.models_dir, f"{model_name}_fold_{fold}.joblib"
                    )
                    model = joblib.load(model_path)
                    fold_preds.append(model.predict_proba(X_test)[:, 1])

                avg_pred = np.mean(fold_preds, axis=0)
                test_level1[model_name] = avg_pred

            else:
                # Load the single fully retrained model
                model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
                model = joblib.load(model_path)
                test_level1[model_name] = model.predict_proba(X_test)[:, 1]

        # Level 2 Prediction
        final_probs = meta_learner.predict_proba(test_level1.values)[:, 1]

        # 6. Save Submission
        print("\n--- Step 6: Saving Submission ---")
        submission = pd.DataFrame(
            {Config.ID_COL: df_test[Config.ID_COL], Config.TARGET_COL: final_probs}
        )

        save_path = os.path.join(self.submission_dir, "submission.csv")
        submission.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
        print("Pipeline Complete.")
