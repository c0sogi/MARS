import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from library.config import Config
from library.utils import seed_everything, mae_metric
from library.feature_extraction import generate_dataset
from library.model_factory import get_base_model, get_meta_model


class StackedEnsemblePipeline:
    def __init__(self):
        # Removed 'cat' due to poor performance (Cite solution_lesson_node_00008)
        self.base_model_names = ["lgbm", "xgb"]
        seed_everything(Config.SEED)

    def load_data(self, debug_size=None, load_cached_data=True):
        """
        Generates/Loads features for train, val, and test sets.
        Combines train and val for CV purposes to maximize data usage.
        """
        # Generate/Load features
        train_df = generate_dataset(
            Config.TRAIN_META_PATH,
            "train_features",
            load_cached_data=load_cached_data,
            debug_size=debug_size,
        )
        val_df = generate_dataset(
            Config.VAL_META_PATH,
            "val_features",
            load_cached_data=load_cached_data,
            debug_size=debug_size,
        )
        test_df = generate_dataset(
            Config.TEST_META_PATH,
            "test_features",
            load_cached_data=load_cached_data,
            debug_size=debug_size,
        )

        # Combine Train and Val for Stacking CV
        full_train_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

        return full_train_df, test_df

    def get_X_y(self, df, is_test=False):
        """
        Separates features, target, and IDs from the dataframe.
        """
        if "segment_id" in df.columns:
            ids = df["segment_id"]
            X = df.drop(columns=["segment_id"])
        else:
            ids = None
            X = df

        if not is_test and "time_to_eruption" in df.columns:
            y = X["time_to_eruption"]
            X = X.drop(columns=["time_to_eruption"])
            return X, y, ids

        # For test set or if target is missing
        if "time_to_eruption" in X.columns:
            X = X.drop(columns=["time_to_eruption"])

        return X, None, ids

    def run_cross_validation(self, X, y, n_folds=5):
        """
        Performs Stratified K-Fold CV to generate OOF predictions.
        """
        print(f"\nRunning {n_folds}-Fold Cross-Validation...")

        # Create bins for stratification of continuous target
        num_bins = 10
        y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

        # Initialize OOF dataframe
        oof_preds = pd.DataFrame(index=X.index)

        # Store scores
        scores = {name: [] for name in self.base_model_names}

        # Arrays to hold OOF predictions for each model
        model_oofs = {name: np.zeros(len(X)) for name in self.base_model_names}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
            print(f"  Fold {fold + 1}/{n_folds}")

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            for name in self.base_model_names:
                model = get_base_model(name)

                # Fit model with Early Stopping
                if name == "lgbm":
                    callbacks = [
                        lgb.early_stopping(
                            stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
                        )
                    ]
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric="mae",
                        callbacks=callbacks,
                    )
                elif name == "xgb":
                    model.fit(
                        X_train, y_train, eval_set=[(X_val, y_val)], verbose=False
                    )
                elif name == "cat":
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=(X_val, y_val),
                        use_best_model=True,
                        verbose=False,
                        early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
                    )

                # Predict
                val_pred = model.predict(X_val)
                model_oofs[name][val_idx] = val_pred

                # Score
                score = mae_metric(y_val, val_pred)
                scores[name].append(score)

        # Aggregate scores
        print("\nCV Results (MAE):")
        for name in self.base_model_names:
            mean_score = np.mean(scores[name])
            print(f"  {name}: {mean_score}")
            oof_preds[f"pred_{name}"] = model_oofs[name]

        return oof_preds

    def train_meta_learner(self, oof_preds, y):
        """
        Trains the Level 1 Meta Learner (Ridge) on OOF predictions.
        """
        print("\nTraining Meta-Learner (Ridge)...")
        meta_model = get_meta_model()
        meta_model.fit(oof_preds, y)

        # Evaluate Meta-Learner on OOF (In-sample for meta, but out-of-sample for base)
        meta_preds = meta_model.predict(oof_preds)
        score = mae_metric(y, meta_preds)
        print(f"  Meta-Learner OOF MAE: {score}")

        return meta_model

    def retrain_base_models(self, X, y):
        """
        Retrains base models on the full dataset.
        Uses a small internal split (10%) for early stopping validation to ensure robustness.
        """
        print("\nRetraining Base Models on Full Dataset...")

        # Split a small portion for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=Config.SEED, shuffle=True
        )

        base_models = {}

        for name in self.base_model_names:
            print(f"  Retraining {name}...")
            model = get_base_model(name)

            if name == "lgbm":
                callbacks = [
                    lgb.early_stopping(
                        stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
                    )
                ]
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="mae",
                    callbacks=callbacks,
                )
            elif name == "xgb":
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            elif name == "cat":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    verbose=False,
                    early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
                )

            base_models[name] = model

        return base_models

    def predict_ensemble(self, base_models, meta_model, X_test):
        """
        Generates predictions using the stacked ensemble.
        """
        print("\nGenerating Ensemble Predictions...")

        # Level 0 predictions
        base_preds = pd.DataFrame(index=X_test.index)
        for name, model in base_models.items():
            base_preds[f"pred_{name}"] = model.predict(X_test)

        # Level 1 predictions
        final_preds = meta_model.predict(base_preds)

        return final_preds

    def run(self, debug_size=None, load_cached_data=True):
        """
        Executes the full pipeline.
        """
        # 1. Load Data
        full_train_df, test_df = self.load_data(
            debug_size=debug_size, load_cached_data=load_cached_data
        )

        X, y, _ = self.get_X_y(full_train_df)
        X_test, _, test_ids = self.get_X_y(test_df, is_test=True)

        # 2. Level 0: Cross-Validation (Generate OOF)
        oof_preds = self.run_cross_validation(X, y, n_folds=Config.N_FOLDS)

        # 3. Level 1: Train Meta-Learner
        meta_model = self.train_meta_learner(oof_preds, y)

        # 4. Retrain Base Models
        base_models = self.retrain_base_models(X, y)

        # 5. Inference
        final_preds = self.predict_ensemble(base_models, meta_model, X_test)

        # 6. Submission
        submission_df = pd.DataFrame(
            {"segment_id": test_ids, "time_to_eruption": final_preds}
        )

        submission_path = Config.SUBMISSION_PATH
        submission_df.to_csv(submission_path, index=False)
        print(f"\nSubmission saved to {submission_path}")
