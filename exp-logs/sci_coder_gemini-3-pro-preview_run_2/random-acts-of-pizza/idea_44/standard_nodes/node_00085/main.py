import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import set_seed, setup_logger, save_submission
from library.data_loader import load_dataset
from library.embedding_generator import generate_embeddings
from library.feature_pipeline import FoldPipeline
from library.model_engine import EnsembleModel

# Initialize Logger
logger = setup_logger("runfile", os.path.join(Config.WORKING_DIR, "runfile.log"))


def get_features_subset(df, embeddings_split, indices=None):
    """
    Helper to slice features (embeddings + metadata) for a specific set of indices.
    """
    # Keys expected by FoldPipeline
    keys_map = {
        "anchor_title": "anchor_title",
        "anchor_body": "anchor_body",
        "aux_title": "aux_title",
        "aux_body": "aux_body",
    }

    subset = {}

    # Slice Embeddings
    for pipeline_key, embed_key in keys_map.items():
        data = embeddings_split[embed_key]
        if indices is not None:
            subset[pipeline_key] = data[indices]
        else:
            subset[pipeline_key] = data

    # Slice Metadata
    if indices is not None:
        subset["meta"] = df.iloc[indices].reset_index(drop=True)
    else:
        subset["meta"] = df.reset_index(drop=True)

    return subset


def main():
    # 1. Setup
    set_seed(Config.SEED)
    logger.info("Starting FS-ADBE Workflow...")

    # 2. Load Data
    # load_dataset returns train/val/test splits based on metadata
    df_train, df_val, df_test = load_dataset(load_cached_data=True)

    # 3. Generate Embeddings
    # Returns dict: {'train': {...}, 'val': {...}, 'test': {...}}
    embeddings = generate_embeddings(df_train, df_val, df_test, load_cached_data=True)

    # 4. Stratified K-Fold Cross-Validation on TRAIN set
    # We train 5 models on the training set to create a robust ensemble.
    n_folds = Config.N_FOLDS
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    y_train_full = df_train["requester_received_pizza"].values.astype(int)

    trained_pipelines = []
    trained_models = []

    logger.info(f"Starting {n_folds}-Fold Cross-Validation on Training Set...")

    for fold_idx, (train_idx, internal_val_idx) in enumerate(
        skf.split(np.zeros(len(y_train_full)), y_train_full)
    ):
        logger.info(f"Processing Fold {fold_idx + 1}/{n_folds}...")

        # Prepare Data for this Fold
        X_fold_train_raw = get_features_subset(df_train, embeddings["train"], train_idx)
        y_fold_train = y_train_full[train_idx]

        # Initialize and Fit Pipeline (PCA, Scaler)
        # Fitting ONLY on fold training data to prevent leakage
        pipeline = FoldPipeline()
        pipeline.fit(X_fold_train_raw)

        # Transform Training Data
        X_fold_train_trans = pipeline.transform(X_fold_train_raw)

        # Initialize and Train Model (Bagged LR with GridSearch)
        model = EnsembleModel()
        model.optimize_and_train(X_fold_train_trans, y_fold_train)

        # Store artifacts
        trained_pipelines.append(pipeline)
        trained_models.append(model)

        # Optional: Save fold artifacts
        pipeline.save(
            os.path.join(Config.WORKING_DIR, f"models/processor_fold_{fold_idx}.joblib")
        )
        model.save(
            os.path.join(Config.WORKING_DIR, f"models/model_fold_{fold_idx}.joblib")
        )

    # 5. Validation on Hold-Out Set
    logger.info("Evaluating on Hold-Out Validation Set...")

    # Prepare Validation Features
    X_val_raw = get_features_subset(df_val, embeddings["val"], indices=None)
    y_val = df_val["requester_received_pizza"].values.astype(int)

    # CV-Bagging Inference: Average predictions from all fold models
    val_preds_accum = np.zeros(len(y_val))

    for i in range(n_folds):
        pipeline = trained_pipelines[i]
        model = trained_models[i]

        # Transform using the specific fold's pipeline
        X_val_trans = pipeline.transform(X_val_raw)

        # Predict
        preds = model.predict_proba(X_val_trans)
        val_preds_accum += preds

    # Average predictions
    val_preds_final = val_preds_accum / n_folds

    # Compute Metric
    val_auc = roc_auc_score(y_val, val_preds_final)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate error magnitude: |y_true - y_pred|
    errors = np.abs(y_val - val_preds_final)

    print("\n--- Failure Analysis: Correlation between Error and Features ---")
    correlations = []
    for col in Config.NUMERICAL_FEATURES:
        if col in df_val.columns:
            feat_values = df_val[col].values
            # Handle potential NaNs just in case, though data loader handles them
            valid_mask = ~np.isnan(feat_values)
            if np.sum(valid_mask) > 1:
                corr, _ = pearsonr(errors[valid_mask], feat_values[valid_mask])
                correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in correlations:
        print(f"{col}: {corr:.4f}")
    print("----------------------------------------------------------------\n")

    # 7. Submission
    THRESHOLD = 0.7201989696216022
    if val_auc > THRESHOLD:
        logger.info(
            f"Validation metric {val_auc} > {THRESHOLD}. Generating submission..."
        )

        # Prepare Test Features
        X_test_raw = get_features_subset(df_test, embeddings["test"], indices=None)
        request_ids = df_test["request_id"].values

        # CV-Bagging Inference on Test
        test_preds_accum = np.zeros(len(request_ids))

        for i in range(n_folds):
            pipeline = trained_pipelines[i]
            model = trained_models[i]

            X_test_trans = pipeline.transform(X_test_raw)
            preds = model.predict_proba(X_test_trans)
            test_preds_accum += preds

        test_preds_final = test_preds_accum / n_folds

        # Save Submission
        save_submission(request_ids, test_preds_final, Config.SUBMISSION_PATH)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        logger.warning(
            f"Validation metric {val_auc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
