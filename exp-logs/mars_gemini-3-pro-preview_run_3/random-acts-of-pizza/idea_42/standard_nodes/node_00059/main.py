import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Import provided library modules
from library import (
    config,
    utils,
    data_loader,
    feature_engineering,
    model_definitions,
    training_pipeline,
)

# Import protected helper for consistent input preparation
from library.training_pipeline import _prepare_model_inputs

# Setup Logging
logger = utils.setup_logging("runfile")


def main():
    # 1. Setup and Reproducibility
    utils.set_seed(config.SEED)
    logger.info("Starting execution...")

    # 2. Load Data
    # We use the full dataset as it is small (~2.3k train, ~0.5k val) and fits easily in memory/time limits.
    # load_cached_data=True ensures we use pre-processed parquet files if available.
    train_df, val_df = data_loader.load_dataset("train", load_cached_data=True)
    test_df = data_loader.load_dataset("test", load_cached_data=True)

    # 3. Feature Extraction
    logger.info("Extracting Static Features...")
    static_extractor = feature_engineering.StaticFeatureExtractor()
    # Extract static features (Embeddings + Metadata)
    static_train = static_extractor.extract(train_df, "train_split")
    static_val = static_extractor.extract(val_df, "val_split")
    static_test = static_extractor.extract(test_df, "test")

    # 4. Level 1 Stacking - OOF Generation
    # This runs CV on train_df to generate inputs for the Meta-Learner
    logger.info("Running Stacking CV on Train Set...")
    stacker = training_pipeline.StackingTrainer()
    oof_df, y_train_oof = stacker.run_cv(train_df, static_train)

    # Train Level 2 Meta-Learner on OOF predictions
    logger.info("Training Meta-Learner on OOF predictions...")
    meta_learner = model_definitions.get_meta_learner()
    meta_learner.fit(oof_df, y_train_oof)

    # 5. Validation Inference
    # To get the "Final Validation Metric" on the hold-out val_df, we must:
    # a) Fit dynamic extractors on train_df
    # b) Train Level 1 models on train_df
    # c) Predict on val_df
    # d) Aggregate with Meta-Learner
    logger.info("Performing Validation Inference...")

    # A. Dynamic Feature Extraction
    dynamic_extractor = feature_engineering.DynamicFeatureExtractor()
    dynamic_extractor.fit(train_df, static_train["metadata"])

    dyn_train = dynamic_extractor.transform(train_df, static_train["metadata"])
    dyn_val = dynamic_extractor.transform(val_df, static_val["metadata"])

    # B. Prepare Inputs
    inputs_train = _prepare_model_inputs(dyn_train, static_train)
    inputs_val = _prepare_model_inputs(dyn_val, static_val)

    # C. Train Level 1 Models
    models_def = model_definitions.get_level1_models()
    val_preds_l1 = np.zeros((len(val_df), len(models_def)))

    y_tr = train_df["requester_received_pizza"].values

    for i, key in enumerate(models_def.keys()):
        model = models_def[key]
        X_tr_full = inputs_train[key]
        X_va = inputs_val[key]

        # Handle XGBoost: Needs Early Stopping to prevent overfitting
        if key == "semantic_xgb":
            # Calculate scale_pos_weight
            n_pos = np.sum(y_tr)
            n_neg = len(y_tr) - n_pos
            scale_weight = n_neg / n_pos if n_pos > 0 else 1.0
            model.set_params(scale_pos_weight=scale_weight)

            # Create an internal split of the training set for Early Stopping
            # We cannot use val_df for ES if we want a fair evaluation on val_df
            tr_idx, es_idx = train_test_split(
                np.arange(len(y_tr)),
                test_size=0.1,
                stratify=y_tr,
                random_state=config.SEED,
            )

            # Slice inputs (handles both CSR sparse and Numpy arrays)
            if hasattr(X_tr_full, "iloc"):
                X_tr, X_es = X_tr_full.iloc[tr_idx], X_tr_full.iloc[es_idx]
            else:
                X_tr, X_es = X_tr_full[tr_idx], X_tr_full[es_idx]

            y_tr_sub, y_es_sub = y_tr[tr_idx], y_tr[es_idx]

            model.fit(X_tr, y_tr_sub, eval_set=[(X_es, y_es_sub)], verbose=False)
        else:
            # RF / Linear models
            model.fit(X_tr_full, y_tr)

        # Predict on Validation Set
        if hasattr(model, "predict_proba"):
            val_preds_l1[:, i] = model.predict_proba(X_va)[:, 1]
        else:
            val_preds_l1[:, i] = model.predict(X_va)

    # D. Meta-Learner Prediction
    val_final_probs = meta_learner.predict_proba(val_preds_l1)[:, 1]

    # E. Metric Calculation
    y_val = val_df["requester_received_pizza"].values
    val_auc = roc_auc_score(y_val, val_final_probs)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis...")
    errors = np.abs(y_val - val_final_probs)

    # Correlate error magnitude with numerical metadata features
    meta_df_val = static_val["metadata"].copy()
    correlations = {}

    for col in meta_df_val.columns:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(meta_df_val[col]):
            # Fill NaNs for correlation calculation
            feat_values = meta_df_val[col].fillna(0)
            # Avoid constant columns
            if feat_values.std() > 0:
                corr = np.corrcoef(feat_values, errors)[0, 1]
                correlations[col] = corr

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for col, corr in sorted_corr[:5]:
        print(f"{col}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.7138293787137718
    if val_auc > THRESHOLD:
        logger.info(
            f"Validation score {val_auc} > {THRESHOLD}. Generating submission..."
        )

        # Use the FinalRetrainer to:
        # 1. Retrain Meta-Learner on OOF (Already done, but FinalRetrainer does it cleanly)
        # 2. Fit Dynamic Extractor on Full Train+Val
        # 3. Retrain Level 1 models on Full Train+Val
        # 4. Predict on Test
        retrainer = training_pipeline.FinalRetrainer()
        retrainer.run(
            train_df,
            val_df,
            test_df,
            static_train,
            static_val,
            static_test,
            oof_df,
            y_train_oof,
        )
    else:
        logger.warning(
            f"Validation score {val_auc} <= {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
