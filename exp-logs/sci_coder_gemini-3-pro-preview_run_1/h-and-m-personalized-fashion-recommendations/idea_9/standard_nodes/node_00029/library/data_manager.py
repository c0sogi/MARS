import pandas as pd
import numpy as np
import os
from library.config import Config


class DataEncoder:
    """
    Handles mapping between raw IDs (strings/ints) and contiguous integer indices.
    Persists mappings to disk to ensure consistency between training and inference.
    """

    def __init__(self):
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

    def fit(self, users, items):
        """
        Create mappings for users and items.
        """
        # Users
        unique_users = np.unique(users)
        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for i, u in enumerate(unique_users)}

        # Items
        unique_items = np.unique(items)
        self.item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx_to_item = {idx: i for idx, i in enumerate(unique_items)}

    def transform_users(self, users):
        """Convert raw user IDs to indices. Returns -1 for unknown users."""
        return np.array([self.user_to_idx.get(u, -1) for u in users], dtype=np.int32)

    def transform_items(self, items):
        """Convert raw item IDs to indices. Returns -1 for unknown items."""
        return np.array([self.item_to_idx.get(i, -1) for i in items], dtype=np.int32)

    def inverse_transform_users(self, indices):
        """Convert indices back to raw user IDs."""
        return [self.idx_to_user.get(i, "UNKNOWN") for i in indices]

    def inverse_transform_items(self, indices):
        """Convert indices back to raw item IDs."""
        return [self.idx_to_item.get(i, "UNKNOWN") for i in indices]

    def save(self, path):
        """Save mappings to parquet files."""
        os.makedirs(path, exist_ok=True)
        pd.DataFrame(
            list(self.user_to_idx.items()), columns=["user", "idx"]
        ).to_parquet(os.path.join(path, "user_map.parquet"), index=False)
        pd.DataFrame(
            list(self.item_to_idx.items()), columns=["item", "idx"]
        ).to_parquet(os.path.join(path, "item_map.parquet"), index=False)

    def load(self, path):
        """Load mappings from parquet files."""
        u_path = os.path.join(path, "user_map.parquet")
        i_path = os.path.join(path, "item_map.parquet")

        if not (os.path.exists(u_path) and os.path.exists(i_path)):
            raise FileNotFoundError(f"Encoder mappings not found in {path}")

        u_df = pd.read_parquet(u_path)
        self.user_to_idx = dict(zip(u_df["user"], u_df["idx"]))
        self.idx_to_user = dict(zip(u_df["idx"], u_df["user"]))

        i_df = pd.read_parquet(i_path)
        self.item_to_idx = dict(zip(i_df["item"], i_df["idx"]))
        self.idx_to_item = dict(zip(i_df["idx"], i_df["item"]))


class DataManager:
    """
    Manages data loading, preprocessing, splitting, and caching for the DWSC model.
    """

    def __init__(self):
        self.encoder = DataEncoder()

    def _load_raw_transactions(self):
        """
        Loads train and validation metadata files and concatenates them.
        """
        # Using optimized dtypes to save memory
        dtypes = {"article_id": "int32", "price": "float32", "sales_channel_id": "int8"}

        print(
            f"Loading raw data from {Config.TRAIN_DATA_PATH} and {Config.VAL_DATA_PATH}..."
        )
        train_df = pd.read_csv(
            Config.TRAIN_DATA_PATH, dtype=dtypes, parse_dates=["t_dat"]
        )
        val_df = pd.read_csv(Config.VAL_DATA_PATH, dtype=dtypes, parse_dates=["t_dat"])

        # Concat to get full history
        df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

        if Config.DEBUG:
            print(
                f"DEBUG mode: Sampling users from the tail to ensure validation overlap."
            )
            # Sample users who appear in the last chunk of data (likely active in target period)
            # We use 2x DEBUG_ROWS to cast a wider net for active users
            tail_users = df.iloc[-Config.DEBUG_ROWS * 2 :]["customer_id"].unique()

            # Select a subset of these users (e.g., 2000) to keep the dataset small but dense
            # Using a fixed seed for reproducibility
            rng = np.random.RandomState(Config.SEED)
            selected_users = rng.choice(
                tail_users, size=min(len(tail_users), 2000), replace=False
            )

            df = df[df["customer_id"].isin(selected_users)].copy()
            print(f"DEBUG: Kept {len(df)} rows for {len(selected_users)} users.")

        return df

    def _fit_encoder(self, df, test_users_df, articles_df):
        """
        Fits the encoder on the union of all observed users and items.
        """
        print("Fitting DataEncoder...")
        all_users = pd.concat(
            [df["customer_id"], test_users_df["customer_id"]]
        ).unique()
        all_items = articles_df["article_id"].unique()
        self.encoder.fit(all_users, all_items)

    def get_validation_data(self, load_cached_data=True):
        """
        Prepares data for validation.
        Split Strategy:
            - Target: Last 7 days of available data.
            - Train: 10 weeks prior to the target period.
        Returns:
            train_df: DataFrame with history (user_id, item_id, days_elapsed)
            target_df: DataFrame with ground truth (user_id, item_id)
            test_users: Array of user_ids to predict for (from target set)
        """
        cache_files = {
            "train": os.path.join(Config.CACHE_DIR, "val_train.parquet"),
            "target": os.path.join(Config.CACHE_DIR, "val_target.parquet"),
            "test_users": os.path.join(Config.CACHE_DIR, "val_test_users.parquet"),
        }

        # Check cache
        if load_cached_data and all(os.path.exists(f) for f in cache_files.values()):
            print("Loading validation data from cache...")
            self.encoder.load(Config.CACHE_DIR)
            train_df = pd.read_parquet(cache_files["train"])
            target_df = pd.read_parquet(cache_files["target"])
            test_users = pd.read_parquet(cache_files["test_users"])["user_id"].values
            return train_df, target_df, test_users

        print("Processing validation data from scratch...")
        Config.setup()

        df = self._load_raw_transactions()
        articles_df = pd.read_csv(Config.ARTICLES_PATH, dtype={"article_id": "int32"})

        # For validation, we need to ensure the encoder covers the sample submission users
        # to maintain consistency, even if we only predict for validation users.
        sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        self._fit_encoder(df, sub_df, articles_df)
        self.encoder.save(Config.CACHE_DIR)

        # Define Split
        max_date = df["t_dat"].max()
        split_date = max_date - pd.Timedelta(days=7)
        print(f"Validation Split Date: {split_date} (Max Date: {max_date})")

        # Target: Last 7 days (split_date < t_dat <= max_date)
        target_mask = df["t_dat"] > split_date
        target_df = df.loc[target_mask].copy()

        # Train: TRAIN_WINDOW_WEEKS ending at split_date
        start_date = split_date - pd.Timedelta(weeks=Config.TRAIN_WINDOW_WEEKS)
        train_mask = (df["t_dat"] <= split_date) & (df["t_dat"] > start_date)
        train_df = df.loc[train_mask].copy()

        print(f"Train range: {train_df['t_dat'].min()} to {train_df['t_dat'].max()}")
        print(f"Target range: {target_df['t_dat'].min()} to {target_df['t_dat'].max()}")

        # Transform IDs
        print("Transforming IDs...")
        train_df["user_id"] = self.encoder.transform_users(train_df["customer_id"])
        train_df["item_id"] = self.encoder.transform_items(train_df["article_id"])
        target_df["user_id"] = self.encoder.transform_users(target_df["customer_id"])
        target_df["item_id"] = self.encoder.transform_items(target_df["article_id"])

        # Calculate days_elapsed relative to split_date (the "now" of validation)
        # days_elapsed = 0 means the transaction happened on split_date
        train_df["days_elapsed"] = (split_date - train_df["t_dat"]).dt.days

        # Filter valid IDs
        train_df = train_df[(train_df["user_id"] != -1) & (train_df["item_id"] != -1)]
        target_df = target_df[
            (target_df["user_id"] != -1) & (target_df["item_id"] != -1)
        ]

        # Test users for validation are those who were active in the target period
        test_users = target_df["user_id"].unique()

        if Config.DEBUG:
            print("DEBUG: Limiting validation test users to 1000.")
            test_users = test_users[:1000]
            target_df = target_df[target_df["user_id"].isin(test_users)]

        # Cache
        print("Caching validation data...")
        train_df.to_parquet(cache_files["train"], index=False)
        target_df.to_parquet(cache_files["target"], index=False)
        pd.DataFrame({"user_id": test_users}).to_parquet(
            cache_files["test_users"], index=False
        )

        return train_df, target_df, test_users

    def get_submission_data(self, load_cached_data=True):
        """
        Prepares data for final submission.
        Split Strategy:
            - Train: Last 10 weeks of available data (up to Max Date).
            - Test Users: All users in sample_submission.csv.
        Returns:
            train_df: DataFrame with history (user_id, item_id, days_elapsed)
            test_users: Array of user_ids to predict for
        """
        cache_files = {
            "train": os.path.join(Config.CACHE_DIR, "sub_train.parquet"),
            "test_users": os.path.join(Config.CACHE_DIR, "sub_test_users.parquet"),
        }

        if load_cached_data and all(os.path.exists(f) for f in cache_files.values()):
            print("Loading submission data from cache...")
            self.encoder.load(Config.CACHE_DIR)
            train_df = pd.read_parquet(cache_files["train"])
            test_users = pd.read_parquet(cache_files["test_users"])["user_id"].values
            return train_df, test_users

        print("Processing submission data from scratch...")
        Config.setup()

        df = self._load_raw_transactions()
        articles_df = pd.read_csv(Config.ARTICLES_PATH, dtype={"article_id": "int32"})
        sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Fit Encoder
        self._fit_encoder(df, sub_df, articles_df)
        self.encoder.save(Config.CACHE_DIR)

        # Define Split
        max_date = df["t_dat"].max()
        print(f"Submission Max Date: {max_date}")

        # Train: TRAIN_WINDOW_WEEKS ending at max_date
        start_date = max_date - pd.Timedelta(weeks=Config.TRAIN_WINDOW_WEEKS)
        train_mask = df["t_dat"] > start_date
        train_df = df.loc[train_mask].copy()

        print(f"Train range: {train_df['t_dat'].min()} to {train_df['t_dat'].max()}")

        # Transform IDs
        print("Transforming IDs...")
        train_df["user_id"] = self.encoder.transform_users(train_df["customer_id"])
        train_df["item_id"] = self.encoder.transform_items(train_df["article_id"])

        # Calculate days_elapsed relative to max_date
        # days_elapsed = 0 means transaction happened on max_date
        train_df["days_elapsed"] = (max_date - train_df["t_dat"]).dt.days

        # Filter valid IDs
        train_df = train_df[(train_df["user_id"] != -1) & (train_df["item_id"] != -1)]

        # Test users are all users in submission file
        sub_df["user_id"] = self.encoder.transform_users(sub_df["customer_id"])
        test_users = sub_df["user_id"].values

        if Config.DEBUG:
            print("DEBUG: Limiting submission test users to 1000.")
            test_users = test_users[:1000]

        # Cache
        print("Caching submission data...")
        train_df.to_parquet(cache_files["train"], index=False)
        pd.DataFrame({"user_id": test_users}).to_parquet(
            cache_files["test_users"], index=False
        )

        return train_df, test_users
