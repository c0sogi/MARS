import pandas as pd
import numpy as np
import os
import gc
from pathlib import Path
from library.config import Config
from library.utils import Timer, reduce_mem_usage


class DataManager:
    """
    Manages data loading, preprocessing, splitting, and caching for the
    Hybrid Multi-Source Retrieval & Interaction-Aware Ranking pipeline.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.input_dir = Config.INPUT_DIR
        self.meta_dir = Config.META_DIR

    def load_data(self, load_cached_data=True):
        """
        Main entry point to get processed data.

        Logic:
        1. Check if processed parquet files exist in working directory.
        2. If yes and load_cached_data is True, load and return them.
        3. If no, load raw data from metadata/input, process (map IDs, split),
           cache them, and return.

        Returns:
            dict: Contains 'train', 'val', 'test', 'articles', 'customers' dataframes
                  and 'article_map', 'customer_map'.
        """
        # Define cache paths
        cache_train = self.working_dir / "processed_train.parquet"
        cache_val = self.working_dir / "processed_val.parquet"
        cache_test = self.working_dir / "processed_test.parquet"
        cache_articles = self.working_dir / "processed_articles.parquet"
        cache_customers = self.working_dir / "processed_customers.parquet"

        # Check if all required cache files exist
        cache_exists = (
            cache_train.exists()
            and cache_val.exists()
            and cache_test.exists()
            and cache_articles.exists()
            and cache_customers.exists()
            and Config.PATH_ARTICLE_MAP.exists()
            and Config.PATH_CUSTOMER_MAP.exists()
        )

        if load_cached_data and cache_exists:
            print("[DataManager] Loading cached data from working directory...")
            with Timer("Load Cached Data"):
                train = pd.read_parquet(cache_train)
                val = pd.read_parquet(cache_val)
                test = pd.read_parquet(cache_test)
                articles = pd.read_parquet(cache_articles)
                customers = pd.read_parquet(cache_customers)
                article_map = pd.read_parquet(Config.PATH_ARTICLE_MAP)
                customer_map = pd.read_parquet(Config.PATH_CUSTOMER_MAP)

            return {
                "train": train,
                "val": val,
                "test": test,
                "articles": articles,
                "customers": customers,
                "article_map": article_map,
                "customer_map": customer_map,
            }

        # If cache missing or reload requested
        print(
            "[DataManager] Cache not found or reload requested. Processing from scratch..."
        )
        return self._process_raw_data()

    def _process_raw_data(self):
        """
        Internal method to load raw files, perform ID mapping, time-splitting, and caching.
        """
        with Timer("Load Raw Metadata"):
            # Load transaction history
            # The metadata folder has train/val split by customer ID.
            # We merge them to perform a strict Time-based split for this specific strategy.
            meta_train = pd.read_parquet(self.meta_dir / "train.parquet")
            meta_val = pd.read_parquet(self.meta_dir / "val.parquet")
            transactions = pd.concat([meta_train, meta_val], ignore_index=True)

            # Load test customers (submission file list)
            test_df = pd.read_parquet(self.meta_dir / "test.parquet")

            # Load entity metadata
            # Force article_id to string to match metadata format (padded)
            articles_df = pd.read_csv(
                self.input_dir / "articles.csv", dtype={"article_id": str}
            )
            customers_df = pd.read_csv(self.input_dir / "customers.csv")

            # Cleanup
            del meta_train, meta_val
            gc.collect()

        with Timer("Preprocessing & ID Mapping"):
            # 1. Standardize article_id format (ensure 10-char padding)
            articles_df["article_id"] = articles_df["article_id"].apply(
                lambda x: x.zfill(10)
            )
            # Standardize transactions article_id to match (padded string)
            transactions["article_id"] = (
                transactions["article_id"].astype(str).apply(lambda x: x.zfill(10))
            )

            # 2. Create Integer Mappings
            # This is crucial for memory efficiency and matrix operations (co-occurrence)

            # Customer Map
            unique_customers = customers_df["customer_id"].unique()
            customer_map = pd.DataFrame(
                {
                    "customer_id": unique_customers,
                    "customer_id_idx": np.arange(len(unique_customers), dtype=np.int32),
                }
            )

            # Article Map
            unique_articles = articles_df["article_id"].unique()
            article_map = pd.DataFrame(
                {
                    "article_id": unique_articles,
                    "article_id_idx": np.arange(len(unique_articles), dtype=np.int32),
                }
            )

            # 3. Apply Mappings
            # Map Customers
            transactions = transactions.merge(
                customer_map, on="customer_id", how="left"
            )
            test_df = test_df.merge(customer_map, on="customer_id", how="left")
            customers_df = customers_df.merge(
                customer_map, on="customer_id", how="left"
            )

            # Map Articles
            transactions = transactions.merge(article_map, on="article_id", how="left")
            articles_df = articles_df.merge(article_map, on="article_id", how="left")

            # 4. Date Conversion
            transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])

            # 5. Memory Optimization
            transactions = reduce_mem_usage(transactions, verbose=False)
            articles_df = reduce_mem_usage(articles_df, verbose=False)
            customers_df = reduce_mem_usage(customers_df, verbose=False)

        with Timer("Time-based Split"):
            # Determine split point: Last 7 days of data
            max_date = transactions["t_dat"].max()
            cutoff_date = max_date - pd.Timedelta(days=Config.VAL_DAYS)

            print(f"  Max Date in Data: {max_date}")
            print(f"  Validation Start (Cutoff): {cutoff_date}")

            # Split
            # Train: <= Cutoff
            # Val: > Cutoff (The last 7 days)
            train_df = transactions[transactions["t_dat"] <= cutoff_date].reset_index(
                drop=True
            )
            val_df = transactions[transactions["t_dat"] > cutoff_date].reset_index(
                drop=True
            )

            print(f"  Train Set Shape: {train_df.shape}")
            print(f"  Val Set Shape: {val_df.shape}")

        with Timer("Caching Artifacts"):
            # Save processed dataframes
            train_df.to_parquet(
                self.working_dir / "processed_train.parquet", index=False
            )
            val_df.to_parquet(self.working_dir / "processed_val.parquet", index=False)
            test_df.to_parquet(self.working_dir / "processed_test.parquet", index=False)
            articles_df.to_parquet(
                self.working_dir / "processed_articles.parquet", index=False
            )
            customers_df.to_parquet(
                self.working_dir / "processed_customers.parquet", index=False
            )

            # Save maps
            article_map.to_parquet(Config.PATH_ARTICLE_MAP, index=False)
            customer_map.to_parquet(Config.PATH_CUSTOMER_MAP, index=False)

        return {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "articles": articles_df,
            "customers": customers_df,
            "article_map": article_map,
            "customer_map": customer_map,
        }

    def get_retrieval_window_data(self, df, weeks=Config.RETRIEVAL_HISTORY_WEEKS):
        """
        Filters the dataframe to keep only the last N weeks of data relative to its max date.
        Used to restrict the history for the Retrieval Stage to recent trends.

        Args:
            df (pd.DataFrame): The transaction dataframe (usually train_df).
            weeks (int): Number of weeks to look back.

        Returns:
            pd.DataFrame: Filtered dataframe.
        """
        if df.empty:
            return df

        max_date = df["t_dat"].max()
        start_date = max_date - pd.Timedelta(weeks=weeks)

        # Filter: Keep transactions strictly after start_date
        filtered_df = df[df["t_dat"] > start_date].copy()
        return filtered_df
