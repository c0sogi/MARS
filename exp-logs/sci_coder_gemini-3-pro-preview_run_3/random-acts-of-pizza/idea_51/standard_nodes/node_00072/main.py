import os
import sys
import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library import config
from library import utils
from library.feature_engineering import FeaturePipeline
from library.engine import EnsembleEngine


def main():
    # 1. Setup and Reproducibility
    utils.set_seed(config.SEED)

    # 2. Feature Generation / Loading
    # The pipeline handles caching. If cache exists in ./working, it loads it.
    # Otherwise it computes features (using GPU for embeddings if available).
    pipeline = FeaturePipeline(load_cached_data=True)
    data = pipeline.run()

    # 3. Model Training (Level 1 & Level 2)
    # Initialize the engine with the prepared data
    engine = EnsembleEngine(data)

    # Run 5-Fold CV for base learners and train the meta-learner
    # This populates engine.oof_matrix and saves models to ./working/models
    engine.run_cv_and_training()

    # Identify Train/Val split for Meta-Learner to prevent leakage
    print("\nPreparing Meta-Learner Training Data...")
    val_meta_path = os.path.join(config.METADATA_DIR, "val.parquet")
    if not os.path.exists(val_meta_path):
        raise FileNotFoundError(f"Validation metadata not found at {val_meta_path}")

    df_val_meta = pd.read_parquet(val_meta_path)
    val_ids_set = set(df_val_meta[config.ID_COL].values)

    # Map validation IDs to indices in the training data
    train_ids = data["train_ids"]
    val_mask = np.isin(train_ids, list(val_ids_set))
    train_mask = ~val_mask

    # Train Meta-Learner on Train subset only (Cite debug_lesson_11)
    engine.train_meta_learner(train_mask=train_mask)

    # 4. Validation Evaluation
    # We must evaluate on the specific hold-out validation set defined in metadata.
    print("\nPerforming Validation Evaluation...")
    if not os.path.exists(val_meta_path):
        # Fallback if metadata structure is different, though instructions guarantee it
        raise FileNotFoundError(f"Validation metadata not found at {val_meta_path}")

    df_val_meta = pd.read_parquet(val_meta_path)
    val_ids_set = set(df_val_meta[config.ID_COL].values)

    # Map validation IDs to indices in the training data
    # data['train_ids'] corresponds to the rows in engine.oof_matrix
    train_ids = data["train_ids"]
    val_mask = np.isin(train_ids, list(val_ids_set))

    if np.sum(val_mask) == 0:
        raise ValueError(
            "Could not locate validation samples within the processed training data."
        )

    # Extract OOF features (predictions from L1 models) for the validation subset
    X_meta_val = engine.oof_matrix.values[val_mask]
    y_val = data["y_train"][val_mask]

    # Load the trained meta-learner to predict on these OOF features
    meta_learner_path = os.path.join(
        config.WORKING_DIR, "models", "meta_learner.joblib"
    )
    meta_learner = joblib.load(meta_learner_path)

    # Get stacked probabilities
    val_preds = meta_learner.predict_proba(X_meta_val)[:, 1]

    # Compute and print the metric
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\nFailure Analysis (Correlation with Error on Validation Set):")
    errors = np.abs(y_val - val_preds)

    # Create a DataFrame for analysis to align features with errors
    df_errors = pd.DataFrame({config.ID_COL: train_ids[val_mask], "error": errors})

    # Merge with the original validation metadata to get interpretable features
    df_analysis = pd.merge(df_errors, df_val_meta, on=config.ID_COL, how="left")

    # Calculate correlations
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns
    correlations = []

    # Exclude non-informative or target columns
    exclude_cols = [
        config.TARGET_COL,
        "error",
        "unix_timestamp_of_request",
        "unix_timestamp_of_request_utc",
    ]

    for col in numeric_cols:
        if col in exclude_cols:
            continue

        # Basic cleaning for correlation calculation
        series = df_analysis[col].fillna(0)
        if series.nunique() < 2:
            continue

        corr, _ = pearsonr(series, df_analysis["error"])
        correlations.append((col, corr))

    # Sort by magnitude of correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error:")
    for feat, corr in correlations[:5]:
        print(f"  {feat}: {corr:.4f}")

    # 6. Submission Generation
    # Strict threshold check
    THRESHOLD = 0.7138293787137718

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        engine.generate_submission()
    else:
        print(
            f"\nValidation metric ({val_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
