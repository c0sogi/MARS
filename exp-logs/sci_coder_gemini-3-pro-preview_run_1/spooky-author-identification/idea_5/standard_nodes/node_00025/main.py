import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import provided library functions
from library.utils import seed_everything, calculate_log_loss, ensure_directory
from library.linear_expert import run_linear_expert
from library.transformer_expert import run_transformer_expert
from library.meta_learner import prepare_stacking_data, XGBoostStacker
from library.feature_engineering import extract_meta_features
from library.data_loader import create_stratified_folds, LABEL_MAP

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    SEED = 42
    N_FOLDS = 5
    DEBUG = False  # Set to False for full training to achieve target score
    THRESHOLD = 0.25336663725445785

    seed_everything(SEED)
    ensure_directory("./submission")

    print("Starting End-to-End Pipeline...")

    # -------------------------------------------------------------------------
    # 2. Level 1 Experts
    # -------------------------------------------------------------------------
    # Run Linear Expert (TF-IDF + Logistic Regression)
    # This is fast and provides stylometric signals.
    print("\n=== Step 1: Running Linear Expert ===")
    run_linear_expert(n_folds=N_FOLDS, seed=SEED, debug=DEBUG, load_cached_data=True)

    # Run Transformer Expert (DeBERTa)
    # This provides deep semantic signals.
    # We use 3 epochs to ensure convergence to beat the threshold.
    print("\n=== Step 2: Running Transformer Expert ===")
    run_transformer_expert(
        n_folds=N_FOLDS,
        seed=SEED,
        debug=DEBUG,
        load_cached_data=True,
        epochs=3,
        batch_size=8,
    )

    # -------------------------------------------------------------------------
    # 3. Level 2 Meta-Learner (Stacking)
    # -------------------------------------------------------------------------
    print("\n=== Step 3: Training Meta-Learner (Stacking) ===")

    # Load aggregated features (Expert OOFs + Meta Features)
    X_train, y_train, X_test, test_ids = prepare_stacking_data(
        n_folds=N_FOLDS, seed=SEED, load_cached_data=True, debug=DEBUG
    )

    # Define XGBoost parameters (aligned with library defaults)
    xgb_params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "n_jobs": -1,
        "verbosity": 0,
    }

    if DEBUG:
        xgb_params["n_estimators"] = 50

    # Initialize and Train Stacker
    stacker = XGBoostStacker(params=xgb_params, n_folds=N_FOLDS, seed=SEED)
    stacker.train(X_train, y_train)

    # -------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Step 4: Validation and Failure Analysis ===")

    # Reconstruct OOF predictions to calculate metric and analyze errors.
    # We iterate through the folds exactly as the stacker did to match validation sets.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds_final = np.zeros((len(X_train), 3))

    # stacker.models contains the trained model for each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_val_fold = X_train[val_idx]
        model = stacker.models[fold]
        # Predict
        val_probs = model.predict_proba(X_val_fold)
        oof_preds_final[val_idx] = val_probs

    # Calculate Final Metric
    final_log_loss = calculate_log_loss(y_train, oof_preds_final)
    print(f"Final Validation Metric: {final_log_loss}")

    # Failure Analysis
    # 1. Calculate per-sample log loss
    # We need to gather the probability assigned to the true class
    # y_train is (N,), oof_preds_final is (N, 3)
    # Clip probs for stability
    eps = 1e-15
    preds_clipped = np.clip(oof_preds_final, eps, 1 - eps)
    # Normalize rows
    preds_clipped /= preds_clipped.sum(axis=1, keepdims=True)

    # Select prob of true class
    rows = np.arange(len(y_train))
    true_class_probs = preds_clipped[rows, y_train]
    sample_losses = -np.log(true_class_probs)

    # 2. Load Meta Features for Correlation
    # We use the helper to get the dataframe
    train_df = create_stratified_folds(
        data_path="./metadata/train.csv",
        n_folds=N_FOLDS,
        seed=SEED,
        load_cached_data=True,
        debug=DEBUG,
    )
    # Ensure alignment: prepare_stacking_data uses create_stratified_folds internally
    # so the order of X_train matches train_df

    meta_df = extract_meta_features(
        train_df, "train_debug" if DEBUG else "train", load_cached_data=True
    )

    analysis_df = pd.DataFrame(
        {
            "loss": sample_losses,
            "char_len": meta_df["char_len"],
            "word_count": meta_df["word_count"],
            "punct_density": meta_df["punct_density"],
        }
    )

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    correlations = analysis_df.corr()["loss"].drop("loss")
    print(correlations)

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    if final_log_loss < THRESHOLD:
        print(
            f"\nValidation Metric ({final_log_loss}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions on test set
        final_test_probs = stacker.predict(X_test)

        # Create submission DataFrame
        submission = pd.DataFrame(final_test_probs, columns=["EAP", "HPL", "MWS"])
        submission.insert(0, "id", test_ids)

        # Save
        sub_path = "./submission/submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"\nValidation Metric ({final_log_loss}) did NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
