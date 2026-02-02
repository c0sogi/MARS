import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import library.config as config


class LGBMRanker:
    def __init__(self, params=None):
        """
        Initializes the LightGBM Ranker.

        Args:
            params (dict): LightGBM hyperparameters. If None, loads from config.
        """
        self.params = params if params is not None else config.LGBM_PARAMS
        self.model = None
        self.feature_cols = None

    def _get_feature_columns(self, df):
        """
        Identifies feature columns by excluding metadata and target columns.
        """
        exclude_cols = {
            config.USER_COL,
            config.ITEM_COL,
            "label",
            "prediction",
            config.DATE_COL,
            "sales_channel_id",  # Often categorical, but if not encoded, exclude.
            # Based on features.py, it's not explicitly added,
            # but we should be safe.
        }
        # We want to include 'retrieval_score', 'rank', and all user/item features
        return [c for c in df.columns if c not in exclude_cols]

    def train(self, df_train, df_val=None, save_model=True):
        """
        Trains the LightGBM ranker using the lambdarank objective.

        Args:
            df_train (pd.DataFrame): Training data with features and 'label'.
            df_val (pd.DataFrame): Validation data with features and 'label'.
            save_model (bool): Whether to save the trained model to disk.
        """
        print("Preparing data for LightGBM training...")

        # 1. Sort by User ID (Required for LambdaRank grouping)
        df_train = df_train.sort_values(by=config.USER_COL).reset_index(drop=True)
        if df_val is not None:
            df_val = df_val.sort_values(by=config.USER_COL).reset_index(drop=True)

        # 2. Identify Features
        self.feature_cols = self._get_feature_columns(df_train)
        print(f"Training with {len(self.feature_cols)} features: {self.feature_cols}")

        # 3. Create Datasets and Groups
        # Group is a list where each element is the number of items for a query (user)
        train_group = df_train.groupby(config.USER_COL).size().values
        train_dataset = lgb.Dataset(
            df_train[self.feature_cols], label=df_train["label"], group=train_group
        )

        valid_sets = [train_dataset]
        valid_names = ["train"]

        if df_val is not None:
            val_group = df_val.groupby(config.USER_COL).size().values
            val_dataset = lgb.Dataset(
                df_val[self.feature_cols],
                label=df_val["label"],
                group=val_group,
                reference=train_dataset,
            )
            valid_sets.append(val_dataset)
            valid_names.append("valid")

        # 4. Train
        print("Starting training...")
        # Note: early_stopping_rounds is passed in callbacks or params depending on version.
        # config.LGBM_PARAMS includes 'early_stopping_rounds'.
        # We use callbacks for recent LightGBM versions compatibility.
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.params.get("early_stopping_rounds", 50)
            ),
            lgb.log_evaluation(period=50),
        ]

        # Remove early_stopping_rounds from params to avoid duplication warning if passed to train
        train_params = self.params.copy()
        if "early_stopping_rounds" in train_params:
            del train_params["early_stopping_rounds"]

        self.model = lgb.train(
            train_params,
            train_dataset,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # 5. Save Model
        if save_model:
            os.makedirs(config.WORKING_DIR, exist_ok=True)
            save_path = config.CACHE_RANKER_MODEL
            print(f"Saving model to {save_path}")
            self.model.save_model(str(save_path))

    def predict(self, df_test, load_cached_model=True):
        """
        Generates predictions for the test set and creates the submission file.

        Args:
            df_test (pd.DataFrame): Test candidates with features.
            load_cached_model (bool): Whether to load the model from disk.
        """
        # 1. Load Model
        if self.model is None:
            if load_cached_model and config.CACHE_RANKER_MODEL.exists():
                print(f"Loading model from {config.CACHE_RANKER_MODEL}")
                self.model = lgb.Booster(model_file=str(config.CACHE_RANKER_MODEL))
            else:
                raise ValueError("Model not trained and no cached model found.")

        # 2. Prepare Features
        # Ensure features match training
        if self.feature_cols is None:
            # Infer from columns, assuming df_test has same structure as train
            self.feature_cols = self._get_feature_columns(df_test)

        print(f"Predicting for {len(df_test)} candidates...")

        # 3. Predict
        preds = self.model.predict(df_test[self.feature_cols])
        df_test = df_test.copy()
        df_test["score"] = preds

        # 4. Select Top 12 per User
        print("Selecting top 12 recommendations per user...")
        # Sort by User and Score (descending)
        df_test.sort_values(
            by=[config.USER_COL, "score"], ascending=[True, False], inplace=True
        )

        # Group and take head
        top_k = df_test.groupby(config.USER_COL).head(12)

        # 5. Format for Submission
        # Convert article_id to string and pad with zeros (if lost during processing)
        # config.ITEM_COL is usually int64.
        top_k["article_id_str"] = top_k[config.ITEM_COL].astype(str).str.zfill(10)

        # Aggregate to space-separated string
        submission_df = (
            top_k.groupby(config.USER_COL)["article_id_str"]
            .apply(" ".join)
            .reset_index()
        )
        submission_df.rename(columns={"article_id_str": "prediction"}, inplace=True)

        # 6. Ensure All Test Users are Present
        print("Ensuring all test users are present...")
        # Load sample submission to get the full list of required users
        sample_sub = pd.read_csv(
            config.SAMPLE_SUBMISSION_CSV, usecols=[config.USER_COL]
        )

        # Merge
        final_sub = sample_sub.merge(submission_df, on=config.USER_COL, how="left")

        # 7. Fallback for Missing Users
        # Identify missing predictions
        missing_mask = final_sub["prediction"].isna()
        n_missing = missing_mask.sum()

        if n_missing > 0:
            print(f"Filling {n_missing} missing predictions with fallback...")
            # Fallback: Top 12 most frequent items in the candidate set
            # This serves as a proxy for 'recent popularity'
            popular_items = df_test[config.ITEM_COL].value_counts().head(12).index
            fallback_str = " ".join([str(x).zfill(10) for x in popular_items])

            final_sub.loc[missing_mask, "prediction"] = fallback_str

        # 8. Save Submission
        print(f"Saving submission to {config.SUBMISSION_PATH}")
        final_sub.to_csv(config.SUBMISSION_PATH, index=False)
