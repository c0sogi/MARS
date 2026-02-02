import os
import pandas as pd
import numpy as np
import datetime
from library.utils import reduce_mem_usage, set_seed


class DataLoader:
    def __init__(
        self,
        input_dir="./input",
        metadata_dir="./metadata",
        cache_dir="./working/idea_2",
    ):
        """
        Initialize the DataLoader.

        Args:
            input_dir (str): Path to the raw input directory.
            metadata_dir (str): Path to the metadata directory.
            cache_dir (str): Path to the directory for storing cached processed data.
        """
        self.input_dir = input_dir
        self.metadata_dir = metadata_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        set_seed(42)

    def load_content_data(self):
        """
        Loads articles and customers data with memory optimization.

        Returns:
            tuple: (articles_df, customers_df)
        """
        print("Loading content data...")
        articles_path = os.path.join(self.input_dir, "articles.csv")
        customers_path = os.path.join(self.input_dir, "customers.csv")

        # Load with specific types where possible to save memory initially
        articles = pd.read_csv(articles_path, dtype={"article_id": "int32"})
        customers = pd.read_csv(customers_path)

        # Apply further memory reduction
        articles = reduce_mem_usage(articles, verbose=False)
        customers = reduce_mem_usage(customers, verbose=False)

        return articles, customers

    def load_transactions(self, load_cached_data=True):
        """
        Loads transactions, merges metadata splits, adds temporal features (week),
        and caches the result.

        Args:
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            pd.DataFrame: The processed transactions dataframe including a 'week' column.
        """
        cache_path = os.path.join(self.cache_dir, "transactions_processed.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading transactions from cache: {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        print("Computing transactions from scratch...")

        train_path = os.path.join(self.metadata_dir, "train.csv")
        val_path = os.path.join(self.metadata_dir, "val.csv")

        # Define types for efficient loading
        dtype_dict = {
            "article_id": "int32",
            "price": "float32",
            "sales_channel_id": "int8",
        }

        # Load the user-split metadata files
        df_train = pd.read_csv(train_path, dtype=dtype_dict)
        df_val = pd.read_csv(val_path, dtype=dtype_dict)

        # Tag the source split (0 for train users, 1 for val users) for potential future use
        df_train["user_split"] = np.int8(0)
        df_val["user_split"] = np.int8(1)

        # Merge to reconstruct full temporal history
        df = pd.concat([df_train, df_val], axis=0, ignore_index=True)

        # Process Dates
        df["t_dat"] = pd.to_datetime(df["t_dat"])

        # Calculate Week Number: 0 is the most recent week, 1 is the week before, etc.
        # This is crucial for the sliding window training strategy.
        max_date = df["t_dat"].max()
        df["week"] = (max_date - df["t_dat"]).dt.days // 7
        df["week"] = df["week"].astype(np.int8)

        # Optimize memory usage
        df = reduce_mem_usage(df)

        # Save to cache
        print(f"Saving processed transactions to cache: {cache_path}")
        df.to_parquet(cache_path, index=False)

        return df

    def get_weekly_split(self, df, target_week):
        """
        Splits the transaction data into history and target sets based on the week index.

        Args:
            df (pd.DataFrame): Transactions dataframe with 'week' column.
            target_week (int): The week index to be used as the target (ground truth).
                               0 = most recent week.

        Returns:
            tuple: (history_df, target_df)
                history_df: Transactions strictly older than target_week.
                target_df: Transactions occurring in target_week.
        """
        # History includes all weeks older than the target week
        history_df = df[df["week"] > target_week].copy()

        # Target is the specific week we are predicting/validating
        target_df = df[df["week"] == target_week].copy()

        return history_df, target_df

    def load_test_customers(self):
        """
        Loads the list of customers required for the submission.

        Returns:
            pd.DataFrame: DataFrame containing 'customer_id' column.
        """
        test_path = os.path.join(self.metadata_dir, "test.csv")
        return pd.read_csv(test_path)
