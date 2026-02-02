import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import Config
from library.utils import Timer, print_memory_usage


class LGBMRanker:
    """
    Stage 2: Dynamic Ensemble Ranking using LightGBM.

    Attributes:
        params (dict): LightGBM hyperparameters.
        model (lgb.Booster): Trained model instance.
        feature_cols (list): List of feature names used for training.
        cat_cols (list): List of categorical feature names.
    """

    def __init__(self):
        self.params = Config.get_lgbm_params()
        self.model = None
        self.feature_cols = []
        self.cat_cols = [
            "product_type_no",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "department_no",
            "index_group_no",
            "section_no",
            "garment_group_no",
            "club_member_status",
            "fashion_news_frequency",
        ]

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Trains the LightGBM ranker using the LambdaRank objective.

        Args:
            train_df (pd.DataFrame): Training data containing features and 'label'.
            val_df (pd.DataFrame): Validation data containing features and 'label'.
        """
        with Timer("LGBM Training Preparation"):
            # 1. Sort data by customer_id (required for LambdaRank groups)
            # We sort in-place or copy? Copy is safer to avoid side effects on input
            train_df = train_df.sort_values("customer_id").reset_index(drop=True)
            val_df = val_df.sort_values("customer_id").reset_index(drop=True)

            # 2. Define Features
            # Exclude ID columns and Label
            exclude_cols = ["customer_id", "article_id", "label", "t_dat"]
            self.feature_cols = [c for c in train_df.columns if c not in exclude_cols]

            print(
                f"Training with {len(self.feature_cols)} features: {self.feature_cols}"
            )

            # 3. Create Query Groups (Group counts for LambdaRank)
            train_group = train_df.groupby("customer_id").size().values
            val_group = val_df.groupby("customer_id").size().values

            # 4. Create LGBM Datasets
            # We construct the dataset with reference to categorical features
            # Filter categorical columns to those actually present in features
            actual_cat_cols = [c for c in self.cat_cols if c in self.feature_cols]

            train_ds = lgb.Dataset(
                train_df[self.feature_cols],
                label=train_df["label"],
                group=train_group,
                categorical_feature=actual_cat_cols,
                free_raw_data=False,
            )

            val_ds = lgb.Dataset(
                val_df[self.feature_cols],
                label=val_df["label"],
                group=val_group,
                categorical_feature=actual_cat_cols,
                reference=train_ds,
                free_raw_data=False,
            )

        # 5. Train
        print("Starting LightGBM training...")
        with Timer("LGBM Training"):
            self.model = lgb.train(
                self.params,
                train_ds,
                valid_sets=[train_ds, val_ds],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
                    lgb.log_evaluation(period=50),
                ],
            )

        # 6. Save Model
        print(f"Saving model to {Config.MODEL_LGBM_FILE}...")
        self.model.save_model(str(Config.MODEL_LGBM_FILE))

        # 7. Feature Importance
        self._print_feature_importance()

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts relevance scores for the candidate set.

        Args:
            test_df (pd.DataFrame): Candidate data with features.

        Returns:
            pd.DataFrame: Original dataframe with an added 'prediction_score' column.
        """
        if self.model is None:
            if Config.MODEL_LGBM_FILE.exists():
                print(f"Loading model from {Config.MODEL_LGBM_FILE}...")
                self.model = lgb.Booster(model_file=str(Config.MODEL_LGBM_FILE))
            else:
                raise FileNotFoundError("Model not found. Call train() first.")

        # Ensure features match
        # If model was loaded, we need to infer features or assume test_df has them
        model_features = self.model.feature_name()

        # Check for missing columns
        missing_cols = set(model_features) - set(test_df.columns)
        if missing_cols:
            raise ValueError(f"Test DataFrame missing features: {missing_cols}")

        print("Predicting scores...")
        with Timer("LGBM Prediction"):
            # Predict
            scores = self.model.predict(test_df[model_features])

        # Return result
        result_df = test_df.copy()
        result_df["prediction_score"] = scores
        return result_df

    def generate_submission(
        self,
        scored_df: pd.DataFrame,
        all_customers_df: pd.DataFrame,
        article_map_path=Config.CACHE_ARTICLE_MAP,
    ):
        """
        Generates the final submission file.

        Args:
            scored_df (pd.DataFrame): Dataframe with 'customer_id', 'article_id', 'prediction_score'.
                                      IDs are mapped integers.
            all_customers_df (pd.DataFrame): Dataframe containing all target 'customer_id' (mapped integers).
            article_map_path (Path): Path to the article ID mapping file.
        """
        print("Generating submission file...")

        # 1. Load Article Map for Inverse Mapping
        article_map = np.load(article_map_path, allow_pickle=True)

        # Helper to convert int ID to string ID
        def get_article_str(idx):
            if 0 <= idx < len(article_map):
                return f"{article_map[idx]:010d}"
            return ""

        # 2. Sort and Rank Candidates
        # Sort by customer and score (descending)
        scored_df = scored_df.sort_values(
            ["customer_id", "prediction_score"], ascending=[True, False]
        )

        # 3. Group and Aggregate
        # We take top 12 per customer
        # Using groupby apply is slow. We can use a faster method since it's sorted.
        # However, for robustness and simplicity with pandas:
        print("Aggregating top 12 predictions...")

        # Filter to top 12
        top_k_df = scored_df.groupby("customer_id").head(Config.TOP_K_PREDICTION)

        # Collect article IDs into a list
        preds = top_k_df.groupby("customer_id")["article_id"].apply(list)

        # 4. Prepare Final DataFrame
        # Ensure we have a row for every customer in the sample submission
        submission_df = pd.DataFrame(
            {"customer_id": all_customers_df["customer_id"].unique()}
        )
        submission_df = submission_df.merge(
            preds.rename("preds"), on="customer_id", how="left"
        )

        # 5. Format Predictions and Handle Fallback
        fallback_str = Config.FALLBACK_PREDICTION
        fallback_list = fallback_str.split()

        # Load Customer Map for inverse mapping
        customer_map = np.load(Config.CACHE_CUSTOMER_MAP, allow_pickle=True)

        results = []

        # Vectorized or list comprehension approach for speed
        # Iterate over rows
        print("Formatting strings and applying fallback...")

        # Convert series to dict for fast lookup
        preds_dict = preds.to_dict()

        # Pre-compute fallback strings to avoid re-doing it
        # But fallback depends on how many items the user already has

        cust_ids = submission_df["customer_id"].values

        # We construct the output list
        # This loop is critical for performance, but python loop over 1.3M is acceptable (~10-20s)

        for cid in cust_ids:
            # Get original customer ID string
            cust_str = customer_map[cid]

            # Get predicted article indices
            pred_indices = preds_dict.get(cid, [])

            # Convert indices to strings
            pred_strs = [get_article_str(idx) for idx in pred_indices]

            # Fill with fallback if needed
            if len(pred_strs) < 12:
                needed = 12 - len(pred_strs)
                pred_strs.extend(fallback_list[:needed])

            # Join
            line = " ".join(pred_strs[:12])
            results.append((cust_str, line))

        # Create final DF
        final_df = pd.DataFrame(results, columns=["customer_id", "prediction"])

        # 6. Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        final_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission generation complete.")

    def _print_feature_importance(self):
        """Prints top feature importance from the trained model."""
        if self.model:
            importance = self.model.feature_importance(importance_type="gain")
            feature_name = self.model.feature_name()

            # Create DataFrame
            df_imp = pd.DataFrame({"feature": feature_name, "importance": importance})
            df_imp = df_imp.sort_values("importance", ascending=False)

            print("\nTop 10 Features by Gain:")
            print(df_imp.head(10))
