import pandas as pd
import numpy as np
import os
from library import settings


class TransactionLoader:
    """
    Handles loading, preprocessing, and splitting of transaction data for the TWIG-SR model.
    Implements time-based windowing, logarithmic temporal decay, and consistent ID mapping.
    """

    def __init__(self):
        self.train_weeks = settings.TRAIN_WEEKS
        self.working_dir = settings.WORKING_DIR
        self.input_dir = settings.INPUT_DIR
        self.metadata_dir = settings.METADATA_DIR

    def get_data(self, validation: bool = False, load_cached_data: bool = True):
        """
        Loads and processes transaction data.

        Parameters
        ----------
        validation : bool
            If True, performs a time-based split (Train: T-10w to T-1w, Val: T).
            If False, uses the full dataset (Train: T-10w to T) for submission.
        load_cached_data : bool
            If True, attempts to load processed data from parquet cache.

        Returns
        -------
        train_df : pd.DataFrame
            Processed training transactions with columns:
            ['user_idx', 'item_idx', 'weight', 'days_elapsed', 't_dat']
        val_df : pd.DataFrame or None
            Validation transactions (ground truth) with columns:
            ['user_idx', 'item_idx']
            None if validation is False.
        user_map : pd.DataFrame
            Mapping from customer_id (str) to user_idx (int).
        item_map : pd.DataFrame
            Mapping from article_id (int) to item_idx (int).
        """
        # Define cache filenames based on mode
        prefix = "val" if validation else "full"
        path_train = os.path.join(self.working_dir, f"{prefix}_train_processed.parquet")
        path_val = os.path.join(self.working_dir, f"{prefix}_val_processed.parquet")
        path_user_map = os.path.join(self.working_dir, f"{prefix}_user_map.parquet")
        path_item_map = os.path.join(self.working_dir, f"{prefix}_item_map.parquet")

        # 1. Try to load from cache
        if load_cached_data:
            files_exist = (
                os.path.exists(path_train)
                and os.path.exists(path_user_map)
                and os.path.exists(path_item_map)
            )
            if validation:
                files_exist = files_exist and os.path.exists(path_val)

            if files_exist:
                print(f"[{prefix.upper()}] Loading data from cache...")
                train_df = pd.read_parquet(path_train)
                user_map = pd.read_parquet(path_user_map)
                item_map = pd.read_parquet(path_item_map)
                val_df = pd.read_parquet(path_val) if validation else None
                return train_df, val_df, user_map, item_map

        # 2. Compute from scratch
        print(f"[{prefix.upper()}] Processing data from scratch...")

        # Load Raw Metadata
        # Use optimized types to save memory
        dtype_dict = {
            "article_id": "int32",
            "price": "float32",
            "sales_channel_id": "int8",
        }

        df_train_meta = pd.read_csv(settings.PATH_TRAIN, dtype=dtype_dict)
        df_val_meta = pd.read_csv(settings.PATH_VAL, dtype=dtype_dict)

        # Convert dates
        df_train_meta["t_dat"] = pd.to_datetime(df_train_meta["t_dat"])
        df_val_meta["t_dat"] = pd.to_datetime(df_val_meta["t_dat"])

        # Combine for global time calculation
        # Note: We must be careful not to leak future data if validation=True,
        # but we need the global max date to establish the relative timeline.
        all_dates = pd.concat([df_train_meta["t_dat"], df_val_meta["t_dat"]])
        max_date = all_dates.max()

        # Define Split Logic
        if validation:
            # Validation Mode:
            # Cutoff is 7 days before the end.
            # Train = All data <= Cutoff (from both user sets to build dense graph)
            # Val = Data > Cutoff (ONLY for users in val set)
            cutoff_date = max_date - pd.Timedelta(days=7)

            # Concatenate all available history for splitting
            full_df = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

            # Split
            mask_history = full_df["t_dat"] <= cutoff_date
            train_raw = full_df[mask_history].copy()

            # For validation target, we only care about the specific validation users
            # identified in metadata/val.csv.
            val_users_set = set(df_val_meta["customer_id"].unique())

            # Get transactions after cutoff
            future_data = full_df[~mask_history]
            # Filter for validation users
            val_raw = future_data[future_data["customer_id"].isin(val_users_set)].copy()

        else:
            # Full Mode:
            # Cutoff is max_date.
            # Train = All data.
            # Val = None.
            cutoff_date = max_date
            train_raw = pd.concat([df_train_meta, df_val_meta], ignore_index=True)
            val_raw = None

        # 3. Apply Time Windowing to Train
        # Keep only last N weeks relative to cutoff_date
        min_date = cutoff_date - pd.Timedelta(weeks=self.train_weeks)
        train_raw = train_raw[train_raw["t_dat"] > min_date].copy()

        print(f"Train Window: {min_date.date()} to {cutoff_date.date()}")
        print(f"Train Rows: {len(train_raw)}")
        if val_raw is not None:
            print(f"Val Rows: {len(val_raw)}")

        # 4. Calculate Weights (Logarithmic Decay)
        # weight = 1 / (1 + log(1 + days_elapsed))
        # days_elapsed = 0 for the most recent day in the cutoff window
        train_raw["days_elapsed"] = (cutoff_date - train_raw["t_dat"]).dt.days

        # Ensure non-negative (sanity check)
        train_raw = train_raw[train_raw["days_elapsed"] >= 0]

        # Apply formula
        # np.log1p(x) is ln(1+x). Formula: 1 / (1 + ln(1 + days))
        train_raw["weight"] = 1.0 / (1.0 + np.log1p(train_raw["days_elapsed"]))
        train_raw["weight"] = train_raw["weight"].astype(settings.FLOAT_DTYPE)

        # 5. Build Mappings
        # We need to map customer_id (str) -> int and article_id (int) -> int (contiguous)

        # Load test users to ensure they are in the map
        df_test = pd.read_csv(settings.PATH_TEST)
        test_users = df_test["customer_id"].unique()

        # Collect all unique users
        unique_users = set(train_raw["customer_id"].unique())
        if val_raw is not None:
            unique_users.update(val_raw["customer_id"].unique())
        unique_users.update(test_users)

        # Create User Map
        user_list = sorted(list(unique_users))
        user_map = pd.DataFrame(
            {
                "customer_id": user_list,
                "user_idx": np.arange(len(user_list), dtype=settings.INT_DTYPE),
            }
        )

        # Collect all unique items (from train only is usually sufficient for CF,
        # but let's include val items to avoid errors during evaluation mapping)
        unique_items = set(train_raw["article_id"].unique())
        if val_raw is not None:
            unique_items.update(val_raw["article_id"].unique())

        # Create Item Map
        item_list = sorted(list(unique_items))
        item_map = pd.DataFrame(
            {
                "article_id": item_list,
                "item_idx": np.arange(len(item_list), dtype=settings.INT_DTYPE),
            }
        )

        # 6. Map DataFrames
        print("Mapping IDs...")

        # Helper to map
        def apply_mapping(df, u_map, i_map):
            # Merge user_idx
            df = df.merge(u_map, on="customer_id", how="inner")
            # Merge item_idx
            df = df.merge(i_map, on="article_id", how="inner")
            return df

        train_final = apply_mapping(train_raw, user_map, item_map)

        # Select columns
        train_cols = ["user_idx", "item_idx", "weight", "days_elapsed", "t_dat"]
        train_final = train_final[train_cols]

        val_final = None
        if val_raw is not None:
            val_final = apply_mapping(val_raw, user_map, item_map)
            val_final = val_final[["user_idx", "item_idx"]]

        # 7. Save to Cache
        print(f"Saving to cache: {self.working_dir}")
        train_final.to_parquet(path_train)
        user_map.to_parquet(path_user_map)
        item_map.to_parquet(path_item_map)

        if val_final is not None:
            val_final.to_parquet(path_val)

        return train_final, val_final, user_map, item_map
