import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import gc
import os

from library.config import Config
from library.utils import setup_logger, set_seed, calculate_accuracy, calculate_log_loss
from library.data_manager import load_data
from library.encoders import MultiClassTargetEncoder

logger = setup_logger("training_engine")


def run_cv_pipeline():
    """
    Executes the Stratified Cross-Validation pipeline with Target Encoding and XGBoost.
    """
    set_seed(Config.SEED)

    # 1. Load Data
    # The data_manager handles caching and basic feature engineering (geometric + dense indices)
    train_df, test_df = load_data(load_cached_data=True)

    # 2. Prepare Target and Features
    if Config.TARGET_COL not in train_df.columns:
        raise ValueError(
            f"Target column {Config.TARGET_COL} missing from training data."
        )

    y = train_df[Config.TARGET_COL]
    # Drop Id and Target from features
    X = train_df.drop(columns=[Config.ID_COL, Config.TARGET_COL], errors="ignore")

    # Prepare Test Features
    test_ids = test_df[Config.ID_COL]
    X_test_base = test_df.drop(columns=[Config.ID_COL], errors="ignore")

    # Encode Labels (XGBoost requires 0..N-1)
    # The dataset classes might be non-contiguous (e.g., 1, 2, 3, 4, 6, 7)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {le.classes_}")

    # 3. Setup Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Arrays to store predictions
    oof_preds = np.zeros((len(train_df), num_classes))
    test_preds_sum = np.zeros((len(test_df), num_classes))

    # Identify columns for Target Encoding
    # We use the dense indices created by data_manager (e.g., "Wilderness_Area_Index")
    cols_to_encode = [f"{prefix}_Index" for prefix in Config.CATEGORICAL_PREFIXES]
    valid_cols_to_encode = [c for c in cols_to_encode if c in X.columns]

    if not valid_cols_to_encode:
        logger.warning("No dense categorical indices found to encode.")
    else:
        logger.info(f"Columns selected for Target Encoding: {valid_cols_to_encode}")

    # Storage for models and test encoding accumulation
    trained_models = []
    test_encoded_accum = None
    encoded_col_names = None

    # 4. CV Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
        logger.info(f"\n--- Starting Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Split Data
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        # --- Feature Engineering: Target Encoding ---
        # We fit on Train, transform Train/Val/Test
        if Config.ENABLE_TARGET_ENCODING and valid_cols_to_encode:
            encoder = MultiClassTargetEncoder(columns=valid_cols_to_encode)
            encoder.fit(X_train, y_train)

            X_train = encoder.transform(X_train)
            X_val = encoder.transform(X_val)

            # Transform Test Set for this fold to accumulate stats
            X_test_fold = encoder.transform(X_test_base)

            # Identify the new columns created by the encoder
            if encoded_col_names is None:
                encoded_col_names = [c for c in X_train.columns if c not in X.columns]
                logger.info(
                    f"Generated {len(encoded_col_names)} target encoded features."
                )

                # Initialize accumulator for test set new columns
                test_encoded_accum = pd.DataFrame(
                    0.0,
                    index=X_test_base.index,
                    columns=encoded_col_names,
                    dtype=np.float32,
                )

            # Accumulate the target encoded values
            test_encoded_accum[encoded_col_names] += X_test_fold[encoded_col_names]

        else:
            # Fallback if encoding is disabled
            X_test_fold = X_test_base.copy()
            encoded_col_names = []

        # --- Model Training ---
        xgb_params = Config.XGB_PARAMS.copy()
        xgb_params["num_class"] = num_classes

        # Create DMatrix for efficiency
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=xgb_params["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=Config.VERBOSE_EVAL,
        )

        # --- Inference on Val ---
        val_preds = model.predict(dval)
        oof_preds[val_idx] = val_preds

        # Calculate Fold Metrics
        fold_acc = calculate_accuracy(y_val, np.argmax(val_preds, axis=1))
        fold_loss = calculate_log_loss(y_val, val_preds)
        logger.info(f"Fold {fold + 1} Accuracy: {fold_acc}")
        logger.info(f"Fold {fold + 1} Log Loss: {fold_loss}")

        # Store model
        trained_models.append(model)

        # Cleanup to free memory
        del X_train, X_val, y_train, y_val, dtrain, dval, X_test_fold
        gc.collect()

    # 5. Global Validation Metrics
    overall_acc = calculate_accuracy(y_encoded, np.argmax(oof_preds, axis=1))
    overall_loss = calculate_log_loss(y_encoded, oof_preds)

    logger.info("\n--- Overall CV Results ---")
    logger.info(f"Overall Accuracy: {overall_acc}")
    logger.info(f"Overall Log Loss: {overall_loss}")

    # 6. Test Inference
    logger.info("\n--- Generating Test Predictions ---")

    # Construct Final Test Features
    # We average the target encodings from all folds to prevent leakage
    X_test_final = X_test_base.copy()
    if encoded_col_names and test_encoded_accum is not None:
        X_test_final[encoded_col_names] = test_encoded_accum / Config.N_FOLDS

    dtest = xgb.DMatrix(X_test_final)

    # Soft Voting
    for i, model in enumerate(trained_models):
        logger.info(f"Predicting with Model {i+1}...")
        preds = model.predict(dtest)
        test_preds_sum += preds

    # Average Probabilities
    avg_test_preds = test_preds_sum / Config.N_FOLDS

    # Convert to Class Labels
    final_preds_indices = np.argmax(avg_test_preds, axis=1)
    final_preds_labels = le.inverse_transform(final_preds_indices)

    # 7. Create Submission
    submission = pd.DataFrame(
        {Config.ID_COL: test_ids, Config.TARGET_COL: final_preds_labels}
    )

    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
