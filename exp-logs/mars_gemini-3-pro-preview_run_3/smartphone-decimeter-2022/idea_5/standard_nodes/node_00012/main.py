import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


# Monkeypatch tqdm to suppress progress bars from libraries
class TqdmNoOp:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable) if self.iterable else iter([])

    def update(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    @classmethod
    def write(cls, *args, **kwargs):
        pass


# Patch modules that might use tqdm
import tqdm

sys.modules["tqdm"].tqdm = TqdmNoOp
# Also patch the specific library module if it imported tqdm directly
import library.postprocess

library.postprocess.tqdm = TqdmNoOp

# Import library modules
from library.config import Config
from library.utils import get_logger, calc_score, haversine_distance
from library.feature_eng import prepare_data
from library.model import LGBMRegressorWrapper
from library.postprocess import apply_postprocessing


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Setup
    set_seed(Config.SEED)
    logger = get_logger("Runfile")

    logger.info("Starting End-to-End Pipeline")

    # 1. Data Loading and Feature Engineering
    # Using load_cached_data=True to use pre-computed parquet files if available
    logger.info("Loading Training Data...")
    train_df = prepare_data("train", load_cached_data=True)

    logger.info("Loading Validation Data...")
    val_df = prepare_data("val", load_cached_data=True)

    logger.info("Loading Test Data...")
    test_df = prepare_data("test", load_cached_data=True)

    # 2. Model Training
    logger.info("Initializing Model...")
    model_wrapper = LGBMRegressorWrapper()

    # Speed optimization: Reduce estimators for fast baseline
    model_wrapper.params["n_estimators"] = 500
    logger.info(
        f"Training with n_estimators={model_wrapper.params['n_estimators']} for speed."
    )

    model_wrapper.train(train_df, val_df)

    # 3. Validation Inference and Post-processing
    logger.info("Running Validation Inference...")
    val_preds_enu = model_wrapper.predict(val_df)

    # Prepare validation dataframe for post-processing (needs metadata)
    # We rely on index alignment; predict returns index-aligned dataframe
    val_for_pp = val_df[
        ["tripId", "UnixTimeMillis", "RefLat", "RefLon", "RefAlt"]
    ].copy()
    val_for_pp[Config.TARGET_EAST] = val_preds_enu[Config.TARGET_EAST]
    val_for_pp[Config.TARGET_NORTH] = val_preds_enu[Config.TARGET_NORTH]

    # Apply coordinate conversion and Kalman Smoothing
    # disable caching here to ensure we process the current model's output
    val_smoothed = apply_postprocessing(val_for_pp, load_cached_data=False)

    # 4. Scoring
    logger.info("Computing Validation Score...")
    # val_smoothed has ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']
    # val_df has the Ground Truth columns
    score = calc_score(val_smoothed, val_df)

    print(f"Final Validation Metric: {score}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Merge predictions with features and GT to analyze errors
    analysis_df = pd.merge(
        val_df,
        val_smoothed[
            ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        ],
        on=["tripId", "UnixTimeMillis"],
        suffixes=("", "_pred"),
    )

    # Calculate error magnitude (meters)
    analysis_df["error_meters"] = haversine_distance(
        analysis_df["LatitudeDegrees"].values,
        analysis_df["LongitudeDegrees"].values,
        analysis_df["LatitudeDegrees_pred"].values,
        analysis_df["LongitudeDegrees_pred"].values,
    )

    # Compute correlations
    feature_cols = Config.FEATURES
    # Ensure features exist in analysis_df (they should be in val_df)
    valid_feats = [f for f in feature_cols if f in analysis_df.columns]

    correlations = (
        analysis_df[valid_feats + ["error_meters"]]
        .corr()["error_meters"]
        .drop("error_meters")
    )
    correlations = correlations.abs().sort_values(ascending=False)

    print("\nTop Feature Correlations with Error Magnitude (Validation):")
    print(correlations.head(10))

    # 6. Submission Generation
    THRESHOLD = 4.32379283550646
    if score < THRESHOLD:
        logger.info(
            f"Validation score {score} passed threshold {THRESHOLD}. Generating submission..."
        )

        # Inference on Test
        test_preds_enu = model_wrapper.predict(test_df)

        # Prepare test dataframe for post-processing
        test_for_pp = test_df[
            ["tripId", "UnixTimeMillis", "RefLat", "RefLon", "RefAlt"]
        ].copy()
        test_for_pp[Config.TARGET_EAST] = test_preds_enu[Config.TARGET_EAST]
        test_for_pp[Config.TARGET_NORTH] = test_preds_enu[Config.TARGET_NORTH]

        # Apply smoothing
        test_smoothed = apply_postprocessing(test_for_pp, load_cached_data=False)

        # Save submission
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        test_smoothed.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")

    else:
        logger.warning(
            f"Validation score {score} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
