import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import gc
import os
import sys

# Import from provided library files
from library.config import Config
from library.utils import setup_logger, set_seed, calculate_accuracy
from library.data_manager import load_data
from library.encoders import MultiClassTargetEncoder


def main():
    # 1. Configuration and Setup
    # Initialize logger (console output)
    logger = setup_logger("runfile")
    # Set reproducible seed
    set_seed(Config.SEED)

    # 2. Load Data
    # Load full dataset with caching enabled for speed
    # This includes geometric feature generation and dense index creation
    train_df, test_df = load_data(load_cached_data=True)

    # 3. Prepare Features and Target
    if Config.TARGET_COL not in train_df.columns:
        raise ValueError(
            f"Target column {Config.TARGET_COL} missing from training data."
        )

    y = train_df[Config.TARGET_COL]
    # Drop Id and Target from features
    X = train_df.drop(columns=[Config.ID_COL, Config.TARGET_COL], errors="ignore")

    # Prepare Test Features (Id needed for submission)
    test_ids = test_df[Config.ID_COL]
    X_test_base = test_df.drop(columns=[Config.ID_COL], errors="ignore")

    # Encode Target Labels (XGBoost requires 0..N-1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Identify columns for Target Encoding (Dense Indices created in data_manager)
    cols_to_encode = [f"{prefix}_Index" for prefix in Config.CATEGORICAL_PREFIXES]
    valid_cols_to_encode = [c for c in cols_to_encode if c in X.columns]

    # 4. Cross-Validation Setup
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Storage for Out-Of-Fold predictions
    oof_preds = np.zeros((len(train_df), num_classes), dtype=np.float32)

    # Storage for Test Set Target Encoding Accumulation
    # We average the target encodings from each fold to prevent leakage and reduce variance
    test_encoded_accum = None
    encoded_col_names = None

    # Store trained models for final ensemble inference
    trained_models = []

    # 5. Training Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
        # Split Data
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        # --- Feature Engineering: Target Encoding ---
        if Config.ENABLE_TARGET_ENCODING and valid_cols_to_encode:
            # Fit only on training data to avoid leakage
            encoder = MultiClassTargetEncoder(columns=valid_cols_to_encode)
            encoder.fit(X_train, y_train)

            # Transform Train and Val
            X_train = encoder.transform(X_train)
            X_val = encoder.transform(X_val)

            # Transform Test Set using this fold's statistics
            X_test_fold = encoder.transform(X_test_base)

            # Initialize accumulator if first fold
            if encoded_col_names is None:
                encoded_col_names = [c for c in X_train.columns if c not in X.columns]
                test_encoded_accum = pd.DataFrame(
                    0.0,
                    index=X_test_base.index,
                    columns=encoded_col_names,
                    dtype=np.float32,
                )

            # Accumulate
            test_encoded_accum[encoded_col_names] += X_test_fold[encoded_col_names]

            # Cleanup temporary fold test data
            del X_test_fold
        else:
            encoded_col_names = []

        # --- Model Training ---
        xgb_params = Config.XGB_PARAMS.copy()
        xgb_params["num_class"] = num_classes

        # Create DMatrix for optimized GPU training
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=xgb_params["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

        # --- Inference on Validation ---
        val_preds = model.predict(dval)
        oof_preds[val_idx] = val_preds

        # Store model
        trained_models.append(model)

        # Cleanup memory
        del X_train, X_val, y_train, y_val, dtrain, dval
        gc.collect()

    # 6. Global Validation Evaluation
    # Convert probabilities to class indices
    final_preds_indices = np.argmax(oof_preds, axis=1)
    overall_acc = calculate_accuracy(y_encoded, final_preds_indices)

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {overall_acc}")

    # 7. Failure Analysis
    # Calculate continuous error magnitude: 1.0 - Probability(True_Class)
    # This quantifies the severity of the error (Cite solution_lesson_node_00022)
    true_class_probs = oof_preds[np.arange(len(y_encoded)), y_encoded]
    error_magnitude = 1.0 - true_class_probs

    # Create a temporary dataframe for correlation analysis
    # We use the original features X (which contains numericals + dense indices + OHE)
    analysis_df = X.copy()
    analysis_df["Error_Magnitude"] = error_magnitude

    # Compute correlation between Error_Magnitude and all features
    correlations = analysis_df.corrwith(analysis_df["Error_Magnitude"]).drop(
        "Error_Magnitude"
    )

    # Sort by absolute correlation strength
    abs_corrs = correlations.abs().sort_values(ascending=False)

    print("\nTop 10 Features Correlated with Error:")
    for feat in abs_corrs.head(10).index:
        corr_val = correlations[feat]
        print(f"{feat}: {corr_val:.6f}")

    # Free memory
    del analysis_df, errors, correlations
    gc.collect()

    # 8. Conditional Submission
    THRESHOLD = 0.9619347222222222

    if overall_acc > THRESHOLD:

        # Construct Final Test Features
        X_test_final = X_test_base.copy()

        # Apply averaged target encodings
        if encoded_col_names and test_encoded_accum is not None:
            X_test_final[encoded_col_names] = test_encoded_accum / Config.N_FOLDS

        dtest = xgb.DMatrix(X_test_final)

        # Soft Voting Ensemble
        test_preds_sum = np.zeros((len(test_df), num_classes), dtype=np.float32)

        for model in trained_models:
            # Predict probabilities
            test_preds_sum += model.predict(dtest)

        # Average probabilities
        avg_test_preds = test_preds_sum / Config.N_FOLDS

        # Convert to Class Labels
        final_test_indices = np.argmax(avg_test_preds, axis=1)
        final_test_labels = le.inverse_transform(final_test_indices)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: final_test_labels}
        )

        # Save
        submission.to_csv(Config.SUBMISSION_FILE, index=False)

    else:
        # Explicitly do not save if threshold not met
        pass


if __name__ == "__main__":
    main()
