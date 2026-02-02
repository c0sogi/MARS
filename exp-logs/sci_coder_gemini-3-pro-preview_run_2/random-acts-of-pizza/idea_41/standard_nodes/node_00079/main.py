import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

# Import from provided library files
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import DataLoader
from library.embedder import EmbeddingGenerator
from library.feature_processor import WhitenedFusionPipeline
from library.model_builder import ModelBuilder

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    logger = setup_logger("RunFile")

    # Optimize for speed as requested (Baseline run)
    # We reduce the number of bagging estimators slightly if needed,
    # but for ~3k rows, 50 estimators is acceptable.
    # We will ensure n_jobs is utilized.

    logger.info(
        "Starting execution of Whitened Multi-Field Asymmetric Dual-Backbone Ensemble..."
    )

    # 2. Data Loading
    # Load cached data if available to save time
    loader = DataLoader()
    train_split_df, val_split_df, test_df = loader.load_data(load_cached=True)

    # 3. Embedding Generation
    # Generates or loads pre-computed embeddings
    embedder = EmbeddingGenerator()
    embeddings = embedder.generate_embeddings(
        train_split_df, val_split_df, test_df, load_cached=True
    )

    # 4. Prepare Data for Cross-Validation
    # Consolidate Train and Val splits into a single Development set for 5-fold CV
    dev_df = pd.concat([train_split_df, val_split_df], axis=0).reset_index(drop=True)
    y = dev_df["requester_received_pizza"].values.astype(int)

    # Stack Embeddings
    dev_anchor = np.vstack([embeddings["train_anchor"], embeddings["val_anchor"]])
    dev_aux = np.vstack([embeddings["train_aux"], embeddings["val_aux"]])

    # Test Embeddings
    test_anchor = embeddings["test_anchor"]
    test_aux = embeddings["test_aux"]

    logger.info(f"Development Set Shape: {dev_df.shape}")
    logger.info(f"Test Set Shape: {test_df.shape}")

    # 5. Cross-Validation Loop
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros(len(dev_df))
    test_preds_accumulator = np.zeros(len(test_df))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(dev_df, y)):
        logger.info(f"--- Fold {fold + 1} / {Config.N_FOLDS} ---")

        # Split Data
        X_train_meta = dev_df.iloc[train_idx].reset_index(drop=True)
        X_val_meta = dev_df.iloc[val_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_val = y[val_idx]

        X_train_anchor = dev_anchor[train_idx]
        X_val_anchor = dev_anchor[val_idx]

        X_train_aux = dev_aux[train_idx]
        X_val_aux = dev_aux[val_idx]

        # Feature Processing Pipeline
        # Fits PCA and Scaler on training fold only
        pipeline = WhitenedFusionPipeline()
        pipeline.fit(X_train_anchor, X_train_aux, X_train_meta)

        # Transform all sets
        X_train_fused = pipeline.transform(X_train_anchor, X_train_aux, X_train_meta)
        X_val_fused = pipeline.transform(X_val_anchor, X_val_aux, X_val_meta)
        X_test_fused = pipeline.transform(test_anchor, test_aux, test_df)

        # Model Training
        builder = ModelBuilder()
        optimizer = builder.get_bagged_lr_optimizer()

        # Fit GridSearchCV
        optimizer.fit(X_train_fused, y_train)

        # Inference
        val_probs = optimizer.predict_proba(X_val_fused)[:, 1]
        oof_preds[val_idx] = val_probs

        # Test Inference (Accumulate)
        test_probs = optimizer.predict_proba(X_test_fused)[:, 1]
        test_preds_accumulator += test_probs

        # Metric
        fold_auc = roc_auc_score(y_val, val_probs)
        fold_scores.append(fold_auc)
        logger.info(f"Fold {fold + 1} AUC: {fold_auc:.6f}")

    # 6. Final Evaluation
    overall_auc = roc_auc_score(y, oof_preds)
    avg_test_preds = test_preds_accumulator / Config.N_FOLDS

    print(f"Final Validation Metric: {overall_auc}")

    # 7. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Create analysis dataframe
    analysis_df = dev_df.copy()
    analysis_df["pred"] = oof_preds
    analysis_df["target"] = y
    analysis_df["error"] = np.abs(analysis_df["target"] - analysis_df["pred"])

    # Select numerical columns for correlation
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target and pred from correlation features
    cols_to_check = [
        c
        for c in numeric_cols
        if c
        not in ["pred", "target", "error", "sample_index", "requester_received_pizza"]
    ]

    correlations = (
        analysis_df[cols_to_check]
        .corrwith(analysis_df["error"])
        .sort_values(key=abs, ascending=False)
    )

    print("\n--- Failure Analysis: Correlation with Error Magnitude ---")
    print(correlations.head(10))
    print("----------------------------------------------------------\n")

    # 8. Conditional Submission
    threshold = 0.7201989696216022
    if overall_auc > threshold:
        logger.info(
            f"Validation metric ({overall_auc}) > threshold ({threshold}). Saving submission."
        )

        submission_path = Config.SUBMISSION_FILE_PATH
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)

        submission_df = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": avg_test_preds,
            }
        )

        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")
    else:
        logger.warning(
            f"Validation metric ({overall_auc}) <= threshold ({threshold}). Submission NOT saved."
        )


if __name__ == "__main__":
    main()
