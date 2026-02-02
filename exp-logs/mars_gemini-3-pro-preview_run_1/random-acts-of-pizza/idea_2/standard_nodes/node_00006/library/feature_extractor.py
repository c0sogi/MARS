import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.impute import SimpleImputer
from library.config import Config
from library.data_manager import DataManager


class HybridFeatureProcessor:
    """
    Responsible for converting raw data into a hybrid numeric representation.
    Generates dense text embeddings and engineers tabular features.
    """

    def __init__(self):
        """
        Initialize the processor with configuration and data manager.
        """
        self.config = Config()
        self.data_manager = DataManager()
        # Imputer for handling missing values in tabular data
        self.imputer = SimpleImputer(strategy="median")

    def _generate_text_embeddings(self, df, model):
        """
        Generates dense vector embeddings for the text content of the requests.
        Combines title and body text.

        Args:
            df (pd.DataFrame): Input dataframe containing text columns.
            model (SentenceTransformer): Pre-trained transformer model.

        Returns:
            pd.DataFrame: DataFrame containing embedding features.
        """
        # Combine title and edit-aware text, handling NaNs
        titles = df["request_title"].fillna("").astype(str)
        texts = df["request_text_edit_aware"].fillna("").astype(str)
        combined_text = (titles + " " + texts).tolist()

        # Generate embeddings
        # show_progress_bar=False to keep output clean
        embeddings = model.encode(combined_text, show_progress_bar=False, batch_size=32)

        # Convert to DataFrame
        embedding_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
        emb_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)

        return emb_df

    def _engineer_tabular_features(self, df):
        """
        Selects numeric columns and creates derived ratio features.
        Note: This function returns the raw numeric features before imputation.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: DataFrame containing numeric features.
        """
        # Identify base numeric columns available in the aligned dataset
        # We exclude the target and text columns
        exclude_cols = [
            "requester_received_pizza",
            "request_title",
            "request_text",
            "request_text_edit_aware",
            "request_id",
            "giver_username_if_known",
            "source_file",
            "requester_username",
            "requester_user_flair",
            "requester_subreddits_at_request",
            "post_was_edited",
        ]

        # Select numeric types
        numeric_df = df.select_dtypes(include=[np.number]).copy()

        # Drop excluded columns if they exist in the numeric selection
        cols_to_drop = [c for c in exclude_cols if c in numeric_df.columns]
        numeric_df = numeric_df.drop(columns=cols_to_drop)

        # --- Feature Engineering ---
        # 1. Interaction Ratios
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-6

        # Ratio of comments to posts (Activity density)
        if (
            "requester_number_of_comments_at_request" in numeric_df.columns
            and "requester_number_of_posts_at_request" in numeric_df.columns
        ):
            numeric_df["feat_comments_per_post"] = numeric_df[
                "requester_number_of_comments_at_request"
            ] / (numeric_df["requester_number_of_posts_at_request"] + 1)

        # Upvotes ratio (Positivity)
        # We have 'plus_downvotes' (sum) and 'minus_downvotes' (diff).
        # sum = up + down, diff = up - down
        # up = (sum + diff) / 2
        # down = (sum - diff) / 2
        if (
            "requester_upvotes_plus_downvotes_at_request" in numeric_df.columns
            and "requester_upvotes_minus_downvotes_at_request" in numeric_df.columns
        ):

            total_votes = numeric_df["requester_upvotes_plus_downvotes_at_request"]
            diff_votes = numeric_df["requester_upvotes_minus_downvotes_at_request"]

            upvotes = (total_votes + diff_votes) / 2
            downvotes = (total_votes - diff_votes) / 2

            numeric_df["feat_upvote_ratio"] = upvotes / (total_votes + epsilon)
            numeric_df["feat_karma_per_day"] = diff_votes / (
                numeric_df.get("requester_account_age_in_days_at_request", 1) + 1
            )

        return numeric_df

    def process_data(self, load_cached_data=True, debug_size=None):
        """
        Main execution method to load, process, and return data.
        Implements caching to avoid re-computing embeddings.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.
            debug_size (int, optional): Number of rows to process for debugging.

        Returns:
            tuple: (train_df, val_df, test_df) processed dataframes.
        """
        # Ensure working directory exists
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

        # Define cache paths
        train_cache = self.config.TRAIN_PROCESSED_PATH
        val_cache = self.config.VAL_PROCESSED_PATH
        test_cache = self.config.TEST_PROCESSED_PATH

        # Check if cache exists and is requested
        if load_cached_data:
            if (
                os.path.exists(train_cache)
                and os.path.exists(val_cache)
                and os.path.exists(test_cache)
            ):
                print("Loading processed data from cache...")
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            else:
                print("Cache not found. Processing from scratch...")

        # 1. Load and Align Data
        print("Loading raw data...")
        raw_train, raw_val, raw_test = self.data_manager.get_data(debug_size=debug_size)

        # 2. Initialize Text Model
        print(
            f"Initializing Sentence Transformer: {self.config.TRANSFORMER_MODEL_NAME}..."
        )
        text_model = SentenceTransformer(self.config.TRANSFORMER_MODEL_NAME)

        # 3. Process Text (Embeddings)
        print("Generating text embeddings...")
        train_emb = self._generate_text_embeddings(raw_train, text_model)
        val_emb = self._generate_text_embeddings(raw_val, text_model)
        test_emb = self._generate_text_embeddings(raw_test, text_model)

        # 4. Process Tabular Data (Feature Engineering)
        print("Engineering tabular features...")
        train_tab = self._engineer_tabular_features(raw_train)
        val_tab = self._engineer_tabular_features(raw_val)
        test_tab = self._engineer_tabular_features(raw_test)

        # 5. Impute Missing Values (Tabular)
        # Fit on Train only to prevent leakage
        print("Imputing missing values...")
        cols = train_tab.columns
        self.imputer.fit(train_tab)

        train_tab_imputed = pd.DataFrame(
            self.imputer.transform(train_tab), columns=cols, index=train_tab.index
        )
        val_tab_imputed = pd.DataFrame(
            self.imputer.transform(val_tab), columns=cols, index=val_tab.index
        )
        test_tab_imputed = pd.DataFrame(
            self.imputer.transform(test_tab), columns=cols, index=test_tab.index
        )

        # 6. Concatenate Features
        # Combine Tabular + Embeddings
        # Also re-attach the target variable and request_id for Train/Val
        print("Concatenating features...")

        def fuse_data(base_df, tab_df, emb_df, is_test=False):
            # Start with processed features
            fused = pd.concat([tab_df, emb_df], axis=1)

            # Add identifiers
            if "request_id" in base_df.columns:
                fused["request_id"] = base_df["request_id"]

            # Add target if available (Train/Val)
            if not is_test and "requester_received_pizza" in base_df.columns:
                # Convert boolean to int (0/1)
                fused["requester_received_pizza"] = base_df[
                    "requester_received_pizza"
                ].astype(int)

            return fused

        train_processed = fuse_data(raw_train, train_tab_imputed, train_emb)
        val_processed = fuse_data(raw_val, val_tab_imputed, val_emb)
        test_processed = fuse_data(raw_test, test_tab_imputed, test_emb, is_test=True)

        # 7. Save to Cache
        print("Saving processed data to cache...")
        train_processed.to_parquet(train_cache, index=False)
        val_processed.to_parquet(val_cache, index=False)
        test_processed.to_parquet(test_cache, index=False)

        return train_processed, val_processed, test_processed
