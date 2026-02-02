import os
import sys
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split

# Import library modules
from library.config import Config
from library.utils import set_seed, setup_logging, timer
from library.data_loader import DataLoader
from library.feature_engine import FeatureEngineer
from library.trainer import StackingTrainer


def main():
    # 1. Initialization
    set_seed(Config.RANDOM_SEED)
    logger = setup_logging()
    logger.info("Starting Hex-View Stacking Ensemble Pipeline")

    # 2. Data Loading
    loader = DataLoader()

    # Load Train, Val, Test
    # We load raw DFs to have access to IDs and raw features for analysis
    train_df = loader.load_data(split="train", load_cached_data=True)
    val_df = loader.load_data(split="val", load_cached_data=True)
    test_df = loader.load_data(split="test", load_cached_data=True)

    # 3. Feature Engineering
    engineer = FeatureEngineer()

    # Fit on Train
    engineer.fit(train_df)

    # Transform all splits
    X_train_dict = engineer.transform(train_df, split="train", load_cached_data=True)
    X_val_dict = engineer.transform(val_df, split="val", load_cached_data=True)
    X_test_dict = engineer.transform(test_df, split="test", load_cached_data=True)

    y_train = X_train_dict["y"]
    y_val = X_val_dict["y"]

    # 4. Training (Stage 1: OOF & Meta-Learner)
    trainer = StackingTrainer()

    # Generate OOF predictions on Train
    oof_train_df = trainer.generate_oof(X_train_dict, y_train)

    # Train Meta-Learner on OOF
    trainer.train_meta(oof_train_df, y_train)

    # 5. Validation (Hold-out Evaluation)
    logger.info("Performing Hold-out Validation...")

    # We need to generate Level 1 predictions for the Validation set.
    # To be strictly unbiased, we must train base models on Train ONLY.
    # Note: trainer.retrain_final_models merges Train+Val for RF/LR, so we can't use it for this step.

    model_names = list(trainer.base_models.keys())
    n_val = len(y_val)
    l1_val_preds = np.zeros((n_val, len(model_names)))

    # Split Train for Early Stopping (for XGBoost models) to avoid using Val for ES during validation check
    # We use a 90/10 split of the Training data.
    tr_idx, es_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=0.1,
        random_state=Config.RANDOM_SEED,
        stratify=y_train,
    )

    for i, name in enumerate(model_names):
        logger.info(f"Validating base model: {name}")
        model = clone(trainer.base_models[name])

        # Prepare full training data for this model
        X_feat_full = trainer._prepare_input(X_train_dict, name)

        # Prepare validation data for this model
        X_val_feat = trainer._prepare_input(X_val_dict, name)

        if "booster" in name:
            # For boosters, we need an eval set for early stopping.
            # We use the 10% split of training data.
            if sp.issparse(X_feat_full):
                X_tr_sub = X_feat_full[tr_idx]
                X_es_sub = X_feat_full[es_idx]
            else:
                X_tr_sub = X_feat_full[tr_idx]
                X_es_sub = X_feat_full[es_idx]

            y_tr_sub = y_train[tr_idx]
            y_es_sub = y_train[es_idx]

            model.fit(
                X_tr_sub, y_tr_sub, eval_set=[(X_es_sub, y_es_sub)], verbose=False
            )
        else:
            # For RF/LR, fit on full training data
            model.fit(X_feat_full, y_train)

        # Predict on Hold-out Validation
        l1_val_preds[:, i] = model.predict_proba(X_val_feat)[:, 1]

    # Level 2 Prediction on Validation
    l1_val_df = pd.DataFrame(l1_val_preds, columns=model_names)
    val_final_probs = trainer.meta_model.predict_proba(l1_val_df)[:, 1]

    # Compute Metric
    val_auc = roc_auc_score(y_val, val_final_probs)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    logger.info("Running Failure Analysis on Validation Set...")

    # Calculate errors
    errors = np.abs(y_val - val_final_probs)

    # Correlate with numerical metadata features
    # We use the processed validation dataframe which has the raw numerical columns
    # We exclude ID and Target
    analysis_cols = val_df.select_dtypes(include=["number"]).columns.tolist()
    exclude_analysis = [Config.ID_COL, Config.TARGET_COL]
    analysis_cols = [c for c in analysis_cols if c not in exclude_analysis]

    correlations = {}
    for col in analysis_cols:
        if col in val_df.columns:
            # Handle potential NaNs just in case, though DataLoader fills them
            feat_vals = val_df[col].fillna(0).values
            # Pearson correlation
            if np.std(feat_vals) > 0 and np.std(errors) > 0:
                corr = np.corrcoef(feat_vals, errors)[0, 1]
                correlations[col] = corr
            else:
                correlations[col] = 0.0

    # Sort and print top correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    logger.info("Top Feature Correlations with Error Magnitude:")
    for feat, corr in sorted_corr[:5]:
        logger.info(f"  {feat}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.7138293787137718

    if val_auc > THRESHOLD:
        logger.info(
            f"Validation score {val_auc} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Retrain Final Models (merging Train + Val where appropriate)
        trainer.retrain_final_models(X_train_dict, y_train, X_val_dict, y_val)

        # Predict on Test
        test_probs = trainer.predict(X_test_dict)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {
                "request_id": test_df[Config.ID_COL],
                "requester_received_pizza": test_probs,
            }
        )

        # Ensure directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.warning(
            f"Validation score {val_auc} did not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
