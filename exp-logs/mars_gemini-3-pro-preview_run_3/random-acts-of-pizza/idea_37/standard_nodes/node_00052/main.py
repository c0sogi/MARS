import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from library import config
from library import utils
from library import data_processing
from library import training_pipeline


def main():
    # 1. Setup and Reproducibility
    utils.set_seed()

    # 2. Load Data
    # We use load_cached_data=True to utilize pre-computed features in ./working
    print("[Runfile] Loading data...")
    X_train, X_val, X_test = data_processing.get_processed_data(load_cached_data=True)

    y_train = X_train["y"]
    y_val = X_val["y"]
    test_ids = X_test["ids"]

    # 3. Initialize Trainer
    trainer = training_pipeline.EnsembleTrainer()

    # 4. Train Meta-Learner via OOF
    # This runs CV on X_train to train the Level 2 Meta-Learner
    trainer.generate_oof(X_train, y_train)

    # 5. Validation Phase (Evaluation on Hold-out Set)
    print("\n[Runfile] Performing validation on hold-out set...")

    # We need to train base models on X_train (Full) to evaluate on X_val.
    # The trainer.base_models currently hold the state from the last CV fold,
    # so we must retrain them properly for validation assessment.

    n_val = len(y_val)
    n_models = len(trainer.model_names)
    val_preds_level1 = np.zeros((n_val, n_models))

    for i, name in enumerate(trainer.model_names):
        model = trainer.base_models[name]

        # Prepare inputs for this specific model view
        X_tr_input = trainer._prepare_input(X_train, name)
        X_val_input = trainer._prepare_input(X_val, name)

        # Fit on Train
        if name == "semantic_booster":
            # XGBoost requires an eval_set for early stopping.
            # To preserve X_val as a strict hold-out for the final metric,
            # we create an internal split from X_train.
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=0.1, random_state=config.RANDOM_STATE
            )
            tr_idx, es_idx = next(sss.split(X_tr_input, y_train))

            X_tr_sub = X_tr_input[tr_idx]
            y_tr_sub = y_train[tr_idx]
            X_es_sub = X_tr_input[es_idx]
            y_es_sub = y_train[es_idx]

            # Recalculate scale_pos_weight for the sub-training set
            neg_count = np.sum(y_tr_sub == 0)
            pos_count = np.sum(y_tr_sub == 1)
            spw = neg_count / pos_count if pos_count > 0 else 1.0
            model.set_params(scale_pos_weight=spw)

            model.fit(
                X_tr_sub, y_tr_sub, eval_set=[(X_es_sub, y_es_sub)], verbose=False
            )
        else:
            # For RF and Linear models, fit on full training set
            model.fit(X_tr_input, y_train)

        # Predict on Validation set
        # Note: predict_proba returns [prob_0, prob_1], we take prob_1
        val_preds_level1[:, i] = model.predict_proba(X_val_input)[:, 1]

    # Generate Final Predictions using the Meta-Learner
    # The meta-learner was trained in step 4 on OOF predictions.
    val_final_probs = trainer.meta_learner.predict_proba(val_preds_level1)[:, 1]

    # Compute Metric
    val_auc = roc_auc_score(y_val, val_final_probs)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\n[Runfile] Performing failure analysis...")
    # Calculate error magnitude
    errors = np.abs(val_final_probs - y_val)

    # Correlate errors with Metadata features
    # Metadata is stored as a dense matrix in the processed dict
    meta_matrix = X_val["metadata"]
    meta_cols = config.METADATA_COLS

    correlations = []
    for idx, col_name in enumerate(meta_cols):
        feat_values = meta_matrix[:, idx]

        # Check for constant values to avoid warnings
        if np.std(feat_values) == 0:
            corr = 0.0
        else:
            corr, _ = stats.pearsonr(feat_values, errors)

        correlations.append((col_name, corr))

    # Sort by magnitude of correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top correlations between Model Error and Metadata Features:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 7. Submission Generation
    threshold = 0.7138293787137718

    if val_auc > threshold:
        print(
            f"\n[Runfile] Validation AUC ({val_auc}) > Threshold ({threshold}). Proceeding to submission."
        )

        # Retrain Base Models on (Train + Val) to maximize performance
        # This method handles the specific retraining logic (e.g. using Val for XGB early stopping)
        trainer.train_final_models(X_train, y_train, X_val, y_val)

        # Generate predictions on Test set
        submission_df = trainer.predict(X_test, test_ids)

        # Save submission
        print(f"Saving submission to {config.SUBMISSION_PATH}")
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")

    else:
        print(
            f"\n[Runfile] Validation AUC ({val_auc}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
