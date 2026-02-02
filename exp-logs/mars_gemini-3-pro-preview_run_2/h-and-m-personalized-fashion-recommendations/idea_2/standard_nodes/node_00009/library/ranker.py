import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

from library.config import (
    LGBM_PARAMS,
    RANKER_FEATURES,
    TARGET_COL,
    MODEL_PATH,
    USER_FEATURES,
    ITEM_FEATURES,
    SEED,
)


class LGBMRanker:
    """
    Stage 2: Ranking Model using LightGBM.
    """

    def __init__(self):
        self.params = LGBM_PARAMS.copy()
        self.features = RANKER_FEATURES
        self.model = None

        # Define categorical features based on config lists
        # We treat ID codes and status columns as categorical
        # 'age', 'cooccurrence_score', 'visual_similarity' are numerical
        self.cat_features = [
            "club_member_status",
            "fashion_news_frequency",
        ] + ITEM_FEATURES

        # Ensure parameters are set correctly
        self.params["verbose"] = -1
        self.params["seed"] = SEED

    def fit(self, train_df, val_df=None, load_cached_data=True):
        """
        Trains the LightGBM ranker.

        Parameters
        ----------
        train_df : pd.DataFrame
            The labeled training dataset.
        val_df : pd.DataFrame, optional
            The labeled validation dataset for early stopping.
        load_cached_data : bool
            If True, attempts to load a pre-trained model from disk.
        """
        # 1. Try Loading from Cache
        if load_cached_data:
            if MODEL_PATH.exists():
                print(f"Loading cached ranker model from {MODEL_PATH}...")
                try:
                    self.model = lgb.Booster(model_file=str(MODEL_PATH))
                    print("Model loaded successfully.")
                    return
                except Exception as e:
                    print(f"Failed to load model: {e}. Retraining...")
            else:
                print("Cached model not found. Training from scratch...")

        print("Preparing LightGBM datasets...")

        # Filter features that are actually in the dataframe
        valid_features = [f for f in self.features if f in train_df.columns]
        valid_cat_features = [f for f in self.cat_features if f in valid_features]

        X_train = train_df[valid_features]
        y_train = train_df[TARGET_COL]

        # Create LightGBM Dataset
        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=valid_cat_features,
            free_raw_data=False,
        )

        valid_sets = [dtrain]
        valid_names = ["train"]

        if val_df is not None:
            X_val = val_df[valid_features]
            y_val = val_df[TARGET_COL]
            dval = lgb.Dataset(
                X_val,
                label=y_val,
                categorical_feature=valid_cat_features,
                reference=dtrain,
                free_raw_data=False,
            )
            valid_sets.append(dval)
            valid_names.append("valid")

        print(f"Training LightGBM with {len(valid_features)} features...")

        # Setup callbacks
        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(
                stopping_rounds=self.params.get("early_stopping_rounds", 50)
            ),
        ]

        # Train
        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Save model
        self.save_model()

        # Print feature importance
        self._print_feature_importance(valid_features)

    def predict(self, test_df):
        """
        Generates scores for candidates.

        Parameters
        ----------
        test_df : pd.DataFrame
            Dataframe containing features.

        Returns
        -------
        np.array
            Predicted probabilities/scores.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Ensure we use the same features as training
        # LightGBM is robust to column order if feature names are used,
        # but we filter to be safe.
        valid_features = [f for f in self.features if f in test_df.columns]

        if not valid_features:
            raise ValueError("No valid features found in test dataframe.")

        return self.model.predict(test_df[valid_features])

    def save_model(self):
        """Saves the trained model to disk."""
        if self.model is None:
            return

        print(f"Saving model to {MODEL_PATH}...")
        # Ensure directory exists
        os.makedirs(MODEL_PATH.parent, exist_ok=True)
        self.model.save_model(str(MODEL_PATH))

    def _print_feature_importance(self, feature_names):
        """Prints top feature importances."""
        if self.model is None:
            return

        importance = self.model.feature_importance(importance_type="gain")
        # Normalize
        importance = importance / importance.sum()

        feature_imp = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        print("\nTop 10 Features by Gain:")
        for _, row in feature_imp.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.6f}")
