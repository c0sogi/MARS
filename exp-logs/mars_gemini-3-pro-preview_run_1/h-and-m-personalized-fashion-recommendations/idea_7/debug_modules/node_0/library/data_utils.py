import pandas as pd
import numpy as np
import os
from datetime import timedelta
from library.config import Config


class Indexer:
    """
    Manages mapping between raw IDs (strings/ints) and dense integer indices.
    Persists mappings using Parquet files to avoid pickle.
    """

    def __init__(self):
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}

    def fit(self, users, items):
        """
        Creates mappings from lists of unique users and items.
        """
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}

    def save(self, path_prefix):
        """
        Saves mappings to parquet files.
        """
        # Convert dicts to DataFrames
        u_df = pd.DataFrame(
            list(self.user_to_idx.items()), columns=["customer_id", "user_idx"]
        )
        i_df = pd.DataFrame(
            list(self.item_to_idx.items()), columns=["article_id", "item_idx"]
        )

        # Save
        u_df.to_parquet(f"{path_prefix}_users.parquet", index=False)
        i_df.to_parquet(f"{path_prefix}_items.parquet", index=False)

    def load(self, path_prefix):
        """
        Loads mappings from parquet files.
        """
        u_df = pd.read_parquet(f"{path_prefix}_users.parquet")
        i_df = pd.read_parquet(f"{path_prefix}_items.parquet")

        self.user_to_idx = dict(zip(u_df["customer_id"], u_df["user_idx"]))
        self.item_to_idx = dict(zip(i_df["article_id"], i_df["item_idx"]))
        self.idx_to_user = dict(zip(u_df["user_idx"], u_df["customer_id"]))
        self.idx_to_item = dict(zip(i_df["item_idx"], i_df["article_id"]))


class DataManager:
    """
    Handles data loading, splitting, preprocessing, and caching.
    """

    def __init__(self):
        self.config = Config
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)
        self.indexer = Indexer()

    def load_data(self, validate=False, load_cached_data=True):
        """
        Main method to load and prepare data.
        Returns:
            train_df, test_df, customers_df, articles_df, indexer
        """
        mode = "val" if validate else "sub"
        cache_prefix = os.path.join(self.config.CACHE_DIR, mode)

        train_path = f"{cache_prefix}_train.parquet"
        test_path = f"{cache_prefix}_test.parquet"
        cust_path = f"{cache_prefix}_customers.parquet"
        art_path = f"{cache_prefix}_articles.parquet"
        idx_prefix = f"{cache_prefix}_index"

        # 1. Try Loading Cache
        if load_cached_data:
            files_exist = (
                os.path.exists(train_path)
                and os.path.exists(test_path)
                and os.path.exists(cust_path)
                and os.path.exists(art_path)
                and os.path.exists(f"{idx_prefix}_users.parquet")
            )

            if files_exist:
                print(f"Loading cached data for {mode} from {self.config.CACHE_DIR}...")
                train_df = pd.read_parquet(train_path)
                test_df = pd.read_parquet(test_path)
                customers = pd.read_parquet(cust_path)
                articles = pd.read_parquet(art_path)
                self.indexer.load(idx_prefix)
                return train_df, test_df, customers, articles, self.indexer

        # 2. Process from Scratch
        print(f"Processing data for {mode} (Cache miss or force reload)...")

        # Load Raw Data
        # Using dtypes to save memory
        trans_df = pd.read_csv(
            os.path.join(self.config.INPUT_DIR, "transactions_train.csv"),
            dtype={
                "article_id": "int32",
                "price": "float32",
                "sales_channel_id": "int8",
            },
            parse_dates=["t_dat"],
        )

        customers = pd.read_csv(os.path.join(self.config.INPUT_DIR, "customers.csv"))
        articles = pd.read_csv(
            os.path.join(self.config.INPUT_DIR, "articles.csv"),
            dtype={"article_id": "int32", "product_code": "int32"},
        )

        # Process Metadata
        customers = self._process_customers(customers)
        articles = self._process_articles(articles)

        # Time Split Logic
        max_date = trans_df["t_dat"].max()

        if validate:
            # Train: [Max-6w, Max-1w), Test: [Max-1w, Max]
            split_date = max_date - timedelta(weeks=self.config.TEST_WEEKS)
            start_date = split_date - timedelta(weeks=self.config.TRAIN_WEEKS)

            train_df = trans_df[
                (trans_df["t_dat"] >= start_date) & (trans_df["t_dat"] < split_date)
            ].copy()
            test_df = trans_df[trans_df["t_dat"] >= split_date].copy()

            # Days elapsed relative to split_date (simulating 'today')
            train_df["days_elapsed"] = (split_date - train_df["t_dat"]).dt.days

        else:
            # Train: [Max-5w, Max], Test: Sample Submission Users
            start_date = max_date - timedelta(weeks=self.config.TRAIN_WEEKS)
            train_df = trans_df[trans_df["t_dat"] >= start_date].copy()

            sub_df = pd.read_csv(
                os.path.join(self.config.INPUT_DIR, "sample_submission.csv")
            )
            test_df = sub_df[["customer_id"]].copy()

            # Days elapsed relative to tomorrow (so max date is 1 day ago)
            ref_date = max_date + timedelta(days=1)
            train_df["days_elapsed"] = (ref_date - train_df["t_dat"]).dt.days

        # Create Indices
        # Fit on all available metadata to ensure consistent mapping
        print("Creating indices...")
        self.indexer.fit(
            customers["customer_id"].unique(), articles["article_id"].unique()
        )

        # Save to Cache
        print("Caching processed data...")
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        customers.to_parquet(cust_path, index=False)
        articles.to_parquet(art_path, index=False)
        self.indexer.save(idx_prefix)

        return train_df, test_df, customers, articles, self.indexer

    def _process_customers(self, df):
        """
        Bins age and handles missing values.
        """
        df["age"] = df["age"].fillna(-1)
        bins = [-2, 0, 18, 25, 35, 45, 55, 65, 100]
        # labels=False returns integer indicators 0, 1, 2...
        df["age_bin"] = pd.cut(df["age"], bins=bins, labels=False).astype(int)
        return df

    def _process_articles(self, df):
        """
        Maps product_code to dense integers for variant matrix.
        """
        # Create dense index for product codes
        unique_codes = df["product_code"].unique()
        code_map = {code: i for i, code in enumerate(unique_codes)}
        df["product_code_idx"] = df["product_code"].map(code_map)
        return df
