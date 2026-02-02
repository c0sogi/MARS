import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import Config
from library.data_utils import seed_everything, load_articles
from library.feature_builder import RankerDatasetGenerator
from library.candidate_retrieval import CandidateRetriever


class LGBMRankerWrapper:
    """
    Wraps the LightGBM Ranker for Stage 2 of the Multi-View Cascade System.
    Handles training with LambdaRank and generating the final submission.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.model_path = Config.RANKER_MODEL_PATH
        self.submission_path = Config.SUBMISSION_PATH
        self.params = Config.LGBM_PARAMS

        # Columns to exclude from features
        self.ignore_cols = {
            "customer_idx",
            "article_idx",
            "label",
            "week",
            "customer_id",
            "article_id",
            "t_dat",
            "prediction",
        }

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        """
        Identifies feature columns from the dataframe.
        """
        return [c for c in df.columns if c not in self.ignore_cols]

    def _prepare_lgb_dataset(
        self, df: pd.DataFrame, feature_cols: list, is_train: bool = True
    ):
        """
        Prepares a LightGBM Dataset with group information for ranking.
        """
        # Sort by query identifiers to ensure correct grouping
        # A query is defined by (week, customer) pair in the training set
        sort_cols = (
            ["week", "customer_idx"] if "week" in df.columns else ["customer_idx"]
        )
        df = df.sort_values(sort_cols)

        # Calculate group sizes (number of candidates per query)
        group_sizes = df.groupby(sort_cols, sort=False).size().values

        X = df[feature_cols]
        y = df["label"] if "label" in df.columns else None

        dataset = lgb.Dataset(
            X,
            label=y,
            group=group_sizes,
            feature_name=feature_cols,
            free_raw_data=False,
        )

        return dataset

    def train(self, load_cached_data: bool = True):
        """
        Trains the LightGBM Ranker using the pre-generated datasets.

        Args:
            load_cached_data (bool): If True, attempts to load a pre-trained model.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # 1. Check Cache
        if load_cached_data and self.model_path.exists():
            print(f"Loading cached LightGBM model from {self.model_path}")
            self.model = lgb.Booster(model_file=str(self.model_path))
            return

        print("Training LightGBM Ranker...")

        # 2. Load Data
        if not Config.RANKER_TRAIN_SET.exists() or not Config.RANKER_VAL_SET.exists():
            raise FileNotFoundError(
                "Ranker datasets not found. Please run feature generation first."
            )

        train_df = pd.read_parquet(Config.RANKER_TRAIN_SET)
        val_df = pd.read_parquet(Config.RANKER_VAL_SET)

        # 3. Prepare Features
        feature_cols = self._get_feature_cols(train_df)
        print(f"Training with {len(feature_cols)} features: {feature_cols}")

        # 4. Create Datasets
        print("Preparing LightGBM Datasets...")
        train_set = self._prepare_lgb_dataset(train_df, feature_cols)
        val_set = self._prepare_lgb_dataset(val_df, feature_cols)

        # Clear memory
        del train_df, val_df
        import gc

        gc.collect()

        # 5. Train
        print("Starting training...")
        evals_result = {}

        self.model = lgb.train(
            self.params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.log_evaluation(period=10),
                lgb.record_evaluation(evals_result),
                lgb.early_stopping(stopping_rounds=self.params["early_stopping_round"]),
            ],
        )

        # Print final metrics with full precision
        if "valid" in evals_result:
            # Assuming ndcg@12 is the first metric
            metric_name = list(evals_result["valid"].keys())[0]
            best_score = evals_result["valid"][metric_name][
                -1
            ]  # Last one recorded (best due to early stopping usually restores best, but lgb.train returns best iteration booster)
            # Actually, let's just print the best score from the booster
            print(f"Best iteration: {self.model.best_iteration}")
            print(
                f"Best Score ({metric_name}): {self.model.best_score['valid'][metric_name]}"
            )

        # 6. Save Model
        print(f"Saving model to {self.model_path}")
        self.model.save_model(str(self.model_path))

    def generate_submission(self, load_cached_data: bool = True):
        """
        Generates the final submission file.

        1. Retrieves candidates for test users.
        2. Enriches candidates with features.
        3. Scores candidates using the trained model.
        4. Selects Top-12 and formats output.
        """
        # Ensure submission directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # 1. Load Model
        if not hasattr(self, "model"):
            if self.model_path.exists():
                print(f"Loading LightGBM model from {self.model_path}")
                self.model = lgb.Booster(model_file=str(self.model_path))
            else:
                raise FileNotFoundError(
                    "Model not found. Please train the model first."
                )

        # 2. Generate Candidates
        print("Generating candidates for submission...")
        retriever = CandidateRetriever()
        candidates_df = retriever.generate_submission_candidates(
            load_cached_data=load_cached_data
        )

        if candidates_df.empty:
            print("Warning: No candidates generated. Creating empty submission.")
            # Handle edge case: create empty submission or fallback to global pop
            # For now, we assume candidates are generated.
            return

        # 3. Feature Engineering
        print("Enriching candidates with features...")
        feat_builder = RankerDatasetGenerator()
        candidates_df = feat_builder.construct_features(candidates_df)

        # 4. Prediction
        print("Scoring candidates...")
        feature_cols = self._get_feature_cols(candidates_df)

        # Check for missing features in test set that were in train
        model_features = self.model.feature_name()
        missing_feats = set(model_features) - set(feature_cols)
        if missing_feats:
            print(
                f"Warning: Missing features in test set: {missing_feats}. Filling with 0."
            )
            for f in missing_feats:
                candidates_df[f] = 0

        # Ensure correct order
        X_test = candidates_df[model_features]
        scores = self.model.predict(X_test)
        candidates_df["pred_score"] = scores

        # 5. Selection (Top 12 per customer)
        print("Selecting Top-12 recommendations...")

        # Sort by customer and score
        candidates_df = candidates_df.sort_values(
            ["customer_idx", "pred_score"], ascending=[True, False]
        )

        # Group and head
        top_candidates = candidates_df.groupby("customer_idx").head(
            Config.RANKING_TOP_K
        )

        # 6. Formatting
        print("Formatting submission...")

        # We need to map article_idx back to article_id (string)
        # And customer_idx back to customer_id (hash)
        # Load maps
        _, article_map = load_articles(load_cached_data=True)
        # Invert map
        idx_to_article = {v: k for k, v in article_map.items()}

        # Map article indices to IDs
        # Use a vectorized map if possible, or apply
        top_candidates["article_id_str"] = top_candidates["article_idx"].map(
            idx_to_article
        )

        # Format article IDs as space-separated string per customer
        # Ensure article_id is string and zfilled if necessary (though map should return original type)
        # The original article_ids are int64, need to convert to 0-padded string
        top_candidates["article_id_str"] = (
            top_candidates["article_id_str"].astype(str).str.zfill(10)
        )

        submission_series = top_candidates.groupby("customer_idx")[
            "article_id_str"
        ].apply(" ".join)

        # 7. Finalize Submission File
        # We need to ensure ALL customers in sample_submission are present
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_CSV)

        # Load customer map to map back customer_idx to customer_id
        # However, we can just merge on customer_id if we have it.
        # top_candidates has customer_idx.
        # We can map submission_series index (customer_idx) back to customer_id
        from library.data_utils import load_customers

        _, customer_map = load_customers(load_cached_data=True)
        idx_to_customer = {v: k for k, v in customer_map.items()}

        submission_df = pd.DataFrame(
            {
                "customer_idx": submission_series.index,
                "prediction": submission_series.values,
            }
        )

        submission_df["customer_id"] = submission_df["customer_idx"].map(
            idx_to_customer
        )

        # Merge with sample submission to include cold-start users (empty predictions)
        final_sub = sample_sub[["customer_id"]].merge(
            submission_df[["customer_id", "prediction"]], on="customer_id", how="left"
        )

        # Fill NaNs with empty string or fallback?
        # Task description: "You must make predictions for all customer_id values"
        # If no candidates, we might want to fill with global popularity (top 12 popular items)
        # Let's check if we have NaNs
        n_missing = final_sub["prediction"].isna().sum()
        if n_missing > 0:
            print(f"Filling {n_missing} missing predictions with Global Popularity...")
            # Load global popularity
            pop_df = feat_builder.beh_builder.build_global_popularity(
                load_cached_data=True
            )
            # Get top 12
            top_pop = pop_df.sort_values("global_popularity", ascending=False).head(12)
            top_pop_ids = [str(aid).zfill(10) for aid in top_pop["article_id"].values]
            pop_str = " ".join(top_pop_ids)

            final_sub["prediction"] = final_sub["prediction"].fillna(pop_str)

        # 8. Save
        print(f"Saving submission to {self.submission_path}")
        final_sub.to_csv(self.submission_path, index=False)
        print("Submission generation complete.")
