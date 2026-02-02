import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from library.config import (
    WORKING_DIR,
    GEOHASH_LEVELS,
    SEED,
    TRAIN_SUBSAMPLE_SIZE,
    XGB_PARAMS,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    SUBMISSION_PATH,
    NUM_FOLDS,
)
from library.data_processor import get_processed_data


class InteractionStatsEngine:
    """
    Engine for computing Multi-Scale Interaction Priors using Dual-Hygiene strategy.
    Leverages the full 'Wisdom' of the dataset (Strict Filter) to provide priors
    for the 'Learner' (Loose Filter) via Vectorized K-Fold Subtraction.
    """

    def __init__(self, working_dir=WORKING_DIR, levels=GEOHASH_LEVELS):
        self.working_dir = working_dir
        self.levels = levels
        self.global_mean = None
        # Cache for stats dataframes: {level: pd.DataFrame}
        self.stats_cache = {}

    def _get_stats_path(self, level):
        return os.path.join(self.working_dir, f"global_route_stats_L{level}.parquet")

    def _get_meta_path(self):
        return os.path.join(self.working_dir, "global_meta.npy")

    def fit(self, df_strict, load_cached=True):
        """
        Computes global statistics (Sum, Count) for each geohash level using the Strict dataset.
        """
        # 1. Compute/Load Global Mean
        meta_path = self._get_meta_path()
        if load_cached and os.path.exists(meta_path):
            self.global_mean = float(np.load(meta_path))
        else:
            self.global_mean = df_strict["fare_amount"].mean()
            np.save(meta_path, np.array(self.global_mean))

        print(f"Global Mean Fare (Strict): {self.global_mean:.4f}")

        # 2. Compute/Load Route Stats per Level
        for level in self.levels:
            stats_path = self._get_stats_path(level)

            if load_cached and os.path.exists(stats_path):
                print(f"Loading cached stats for Level {level}...")
                self.stats_cache[level] = pd.read_parquet(stats_path)
            else:
                print(f"Computing global stats for Level {level}...")
                # Group by Route
                route_cols = [f"pickup_geohash_{level}", f"dropoff_geohash_{level}"]

                # Aggregation
                # We use float32 for sums to save space, int32 for counts
                stats = (
                    df_strict.groupby(route_cols)["fare_amount"]
                    .agg(["sum", "count"])
                    .reset_index()
                )
                stats.columns = route_cols + [f"sum_L{level}", f"count_L{level}"]

                # Optimize types
                stats[f"sum_L{level}"] = stats[f"sum_L{level}"].astype(np.float32)
                stats[f"count_L{level}"] = stats[f"count_L{level}"].astype(np.int32)

                # Save
                stats.to_parquet(stats_path, index=False)
                self.stats_cache[level] = stats

                # Clean up to save memory during loop
                del stats
                gc.collect()

                # Reload to ensure consistency
                self.stats_cache[level] = pd.read_parquet(stats_path)

    def transform_train(self, df_train, num_folds=NUM_FOLDS):
        """
        Applies Vectorized K-Fold Subtraction to generate interaction features for training.
        Formula: Mean_Rest = (Global_Sum - Fold_Sum) / (Global_Count - Fold_Count)
        """
        print(
            "Generating interaction features for Training (Vectorized Subtraction)..."
        )
        df_out = df_train.copy()

        # Assign folds
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=SEED)
        df_out["fold_id"] = -1
        # Use a temporary array for fold assignment to avoid fragmentation
        fold_ids = np.full(len(df_out), -1, dtype=np.int32)
        for fold_idx, (_, val_idx) in enumerate(kf.split(df_out)):
            fold_ids[val_idx] = fold_idx
        df_out["fold_id"] = fold_ids

        for level in self.levels:
            print(f"  Processing Level {level}...")
            route_cols = [f"pickup_geohash_{level}", f"dropoff_geohash_{level}"]
            global_stats = self.stats_cache[level]

            # 1. Merge Global Stats
            # Left join: preserve train rows. If not in global, gets NaN.
            df_out = df_out.merge(global_stats, on=route_cols, how="left")

            # 2. Compute Fold Stats (for the specific rows in this dataset)
            # Group by [Route, Fold]
            fold_group_cols = route_cols + ["fold_id"]
            fold_stats = (
                df_out.groupby(fold_group_cols)["fare_amount"]
                .agg(["sum", "count"])
                .reset_index()
            )
            fold_stats.columns = fold_group_cols + ["fold_sum", "fold_count"]

            # 3. Merge Fold Stats back
            df_out = df_out.merge(fold_stats, on=fold_group_cols, how="left")

            # 4. Fill NaNs for Fold Stats (if no rows in fold, sum/count is 0)
            df_out["fold_sum"] = df_out["fold_sum"].fillna(0)
            df_out["fold_count"] = df_out["fold_count"].fillna(0)

            # 5. Vectorized Subtraction
            # Rest = Global - Fold
            g_sum_col = f"sum_L{level}"
            g_cnt_col = f"count_L{level}"

            # If global is NaN (route not in strict), Rest is NaN.
            rest_sum = df_out[g_sum_col] - df_out["fold_sum"]
            rest_count = df_out[g_cnt_col] - df_out["fold_count"]

            # 6. Compute Mean
            mean_col = f"mean_fare_L{level}"
            df_out[mean_col] = rest_sum / rest_count

            # 7. Fallback / Cleanup
            # If rest_count <= 0, we have no prior. Leave as NaN (XGBoost handles it).
            mask_invalid = rest_count <= 0
            df_out.loc[mask_invalid, mean_col] = np.nan

            # Drop temp columns to free memory
            df_out.drop(
                columns=[g_sum_col, g_cnt_col, "fold_sum", "fold_count"], inplace=True
            )

        # Drop fold_id
        df_out.drop(columns=["fold_id"], inplace=True)
        return df_out

    def transform_test(self, df_test):
        """
        Applies Global Stats directly to Test/Validation data.
        """
        print("Generating interaction features for Test/Val (Direct Mapping)...")
        df_out = df_test.copy()

        for level in self.levels:
            route_cols = [f"pickup_geohash_{level}", f"dropoff_geohash_{level}"]
            global_stats = self.stats_cache[level]

            # Merge Global Stats
            df_out = df_out.merge(global_stats, on=route_cols, how="left")

            # Compute Mean
            g_sum_col = f"sum_L{level}"
            g_cnt_col = f"count_L{level}"
            mean_col = f"mean_fare_L{level}"

            df_out[mean_col] = df_out[g_sum_col] / df_out[g_cnt_col]

            # Drop temp columns
            df_out.drop(columns=[g_sum_col, g_cnt_col], inplace=True)

        return df_out


def train_and_predict(
    subsample_size=TRAIN_SUBSAMPLE_SIZE,
    num_boost_round=NUM_BOOST_ROUND,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
):
    """
    Executes the full pipeline:
    1. Load Strict Data -> Fit Stats Engine
    2. Load Train Subsample -> Transform (Vectorized Subtraction)
    3. Load Val -> Transform (Direct)
    4. Train XGBoost
    5. Load Test -> Transform (Direct) -> Predict
    6. Save Submission
    """

    # --- Step 1: Wisdom Phase (Global Stats Generation) ---
    print("\n=== Phase 1: Wisdom (Global Stats Generation) ===")
    # Load strict data (cached if available)
    df_strict = get_processed_data("train", mode="strict", load_cached_data=True)

    # Initialize and Fit Engine
    engine = InteractionStatsEngine()
    engine.fit(df_strict, load_cached=True)

    # Free memory immediately
    del df_strict
    gc.collect()

    # --- Step 2: Learner Phase (Training Data Prep) ---
    print("\n=== Phase 2: Learner (Training Data Prep) ===")
    df_train = get_processed_data(
        "train", mode="loose", subsample_size=subsample_size, load_cached_data=True
    )
    df_train = engine.transform_train(df_train)

    # Define features
    exclude_cols = {"key", "fare_amount", "pickup_datetime", "fold_id"}
    # Also exclude geohash strings from features passed to XGBoost
    exclude_cols.update([c for c in df_train.columns if "geohash" in c])

    features = [c for c in df_train.columns if c not in exclude_cols]
    target = "fare_amount"

    print(f"Training Features ({len(features)}): {features}")

    # --- Step 3: Validation Data ---
    print("\n=== Phase 3: Validation Data Prep ===")
    # Use loose filtering for validation to match training distribution but keep it reasonably clean
    df_val = get_processed_data("val", mode="loose", load_cached_data=True)
    df_val = engine.transform_test(df_val)

    # --- Step 4: Model Training ---
    print("\n=== Phase 4: Model Training ===")
    X_train = df_train[features]
    y_train = df_train[target]
    X_val = df_val[features]
    y_val = df_val[target]

    print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=VERBOSE_EVAL,
    )

    # Free memory
    del df_train, df_val, X_train, y_train, X_val, y_val, dtrain, dval
    gc.collect()

    # --- Step 5: Prediction ---
    print("\n=== Phase 5: Prediction ===")
    # Test data uses 'inference' mode (no filtering)
    df_test = get_processed_data("test", mode="inference", load_cached_data=True)
    df_test = engine.transform_test(df_test)

    X_test = df_test[features]
    dtest = xgb.DMatrix(X_test)

    preds = model.predict(dtest)

    # Post-processing: Floor at $2.50
    preds = np.maximum(preds, 2.50)

    # Save Submission
    submission = pd.DataFrame({"key": df_test["key"], "fare_amount": preds})

    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Pipeline Complete.")
