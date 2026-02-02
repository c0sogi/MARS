import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.data_factory import DataFactory
from library.ensemble_runner import EnsembleRunner

# --- 1. Configuration for Fast Baseline ---
# We modify the Config class attributes directly to speed up training
# while maintaining the structural integrity of the ensemble.
Config.XGB_FIT_PARAMS["num_boost_round"] = 300
Config.XGB_FIT_PARAMS["early_stopping_rounds"] = 30
Config.NN_PARAMS["epochs"] = 5
Config.NN_PARAMS["batch_size"] = 4096
Config.NN_PARAMS["hidden_dims"] = [128, 64]  # Reduced complexity for speed
Config.NN_PARAMS["patience"] = 2

# Subsampling fraction for the fast baseline
SUBSAMPLE_FRAC = 0.20


class ValidationRunner(EnsembleRunner):
    """
    Runner for the Validation Phase.
    Trains on a subsample of Train.
    Predicts on the full Validation set (treated as Test).
    """

    def _get_data(self):
        print("Loading data for Validation Run...")
        # Load engineered data
        train_df, val_df, test_df, test_ids = DataFactory.load_and_engineer_data(
            load_cached_data=True
        )

        # Subsample Train for speed
        train_df = train_df.sample(
            frac=SUBSAMPLE_FRAC, random_state=Config.SEED
        ).reset_index(drop=True)
        print(f"Subsampled Training Data: {train_df.shape}")

        # Prepare X, y from Train
        y = (train_df[Config.TARGET_COL] - 1).values.astype(np.int64)
        X = train_df.drop(columns=[Config.TARGET_COL]).values.astype(np.float32)

        # Prepare 'Test' from Val (Full Val set for accurate metric)
        # Note: val_df has features engineered and ID dropped (if present)
        X_test = val_df.drop(columns=[Config.TARGET_COL]).values.astype(np.float32)

        # Create dummy IDs for the validation output file
        test_ids = np.arange(len(val_df))

        return X, y, X_test, test_ids


class SubmissionRunner(EnsembleRunner):
    """
    Runner for the Submission Phase.
    Trains on a subsample of Full Train (Train + Val).
    Predicts on the actual Test set.
    """

    def _get_data(self):
        print("Loading data for Submission Run...")
        train_df, val_df, test_df, test_ids = DataFactory.load_and_engineer_data(
            load_cached_data=True
        )

        # Concat Train + Val
        full_train = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

        # Subsample for speed
        full_train = full_train.sample(
            frac=SUBSAMPLE_FRAC, random_state=Config.SEED
        ).reset_index(drop=True)
        print(f"Subsampled Full Training Data: {full_train.shape}")

        y = (full_train[Config.TARGET_COL] - 1).values.astype(np.int64)
        X = full_train.drop(columns=[Config.TARGET_COL]).values.astype(np.float32)

        X_test = test_df.values.astype(np.float32)

        return X, y, X_test, test_ids


def main():
    seed_everything(Config.SEED)

    # --- Step 1: Validation Phase ---
    print("\n>>> STARTING VALIDATION PHASE <<<")

    # Set temporary submission path for validation predictions
    val_preds_path = os.path.join(Config.WORKING_DIR, "val_predictions.csv")
    Config.SUBMISSION_PATH = val_preds_path

    # Run Ensemble
    val_runner = ValidationRunner()
    val_runner.run_kfold_stacking()

    # --- Step 2: Compute Metric ---
    print("\n>>> COMPUTING VALIDATION METRICS <<<")

    # Load predictions
    if not os.path.exists(val_preds_path):
        raise FileNotFoundError("Validation predictions not found.")

    val_preds_df = pd.read_csv(val_preds_path)
    preds = val_preds_df[Config.TARGET_COL].values

    # Load True Targets from Metadata
    # We rely on the fact that DataFactory reads the CSV sequentially and preserves order
    val_meta = pd.read_csv(Config.VAL_PATH)
    targets = val_meta[Config.TARGET_COL].values

    if len(preds) != len(targets):
        raise ValueError(
            f"Length mismatch: Preds {len(preds)} vs Targets {len(targets)}"
        )

    acc = np.mean(preds == targets)
    print(f"Final Validation Metric: {acc}")

    # --- Step 3: Failure Analysis ---
    print("\n>>> PERFORMING FAILURE ANALYSIS <<<")

    # Create analysis dataframe
    analysis_df = val_meta.copy()
    analysis_df["Predicted"] = preds
    analysis_df["Error"] = (
        analysis_df[Config.TARGET_COL] != analysis_df["Predicted"]
    ).astype(int)

    # Compute correlation of Error with numerical features
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
    correlations = []

    for col in numeric_cols:
        if col not in ["Error", "Predicted", "Id", "Cover_Type"]:
            # Handle potential NaNs just in case, though data should be clean
            if analysis_df[col].isnull().sum() == 0:
                corr = analysis_df["Error"].corr(analysis_df[col])
                correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error:")
    for name, val in correlations[:5]:
        print(f"{name}: {val:.4f}")

    # --- Step 4 & 5: Submission Phase ---
    THRESHOLD = 0.9614708333333334

    if acc > THRESHOLD:
        print("\n>>> VALIDATION PASSED. GENERATING SUBMISSION <<<")

        # Reset Submission Path to the required output location
        Config.SUBMISSION_PATH = "./submission/submission.csv"

        # Run Ensemble on Test Set
        sub_runner = SubmissionRunner()
        sub_runner.run_kfold_stacking()

    else:
        print(
            f"\n>>> VALIDATION FAILED ({acc} <= {THRESHOLD}). SKIPPING SUBMISSION <<<"
        )


if __name__ == "__main__":
    main()
