import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Import from provided library files
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_manager import DataManager
from library.trainer import Trainer


def main():
    # 1. Setup and Configuration
    set_seed(Config.RANDOM_SEED)

    # Suppress library logs to meet "Only print the required information" constraint
    # We set them to WARNING so we don't see INFO logs from the training process
    logging.getLogger("trainer").setLevel(logging.WARNING)
    logging.getLogger("data_manager").setLevel(logging.WARNING)
    logging.getLogger("feature_engine").setLevel(logging.WARNING)
    logging.getLogger("model_factory").setLevel(logging.WARNING)
    logging.getLogger("execution").setLevel(logging.WARNING)

    # 2. Run Training Pipeline
    # This generates embeddings, trains models, saves artifacts, and creates the initial submission
    trainer = Trainer()
    trainer.run_cross_validation(load_cached_data=True)

    # 3. Reconstruct OOF Predictions for Metric & Analysis
    # We need to reload data and artifacts to exactly reproduce the OOF state
    # because Trainer doesn't return them directly.

    data_manager = DataManager()
    df_train = data_manager.load_dataset(split="train", load_cached_data=True)
    target = df_train[Config.TARGET_COL].values

    # Load pre-computed training embeddings
    if os.path.exists(Config.TRAIN_EMBEDDINGS_PATH):
        train_text_emb = np.load(Config.TRAIN_EMBEDDINGS_PATH)
    else:
        # Fallback if cache missing (unlikely given trainer just ran)
        from library.feature_engine import TextEmbedder

        embedder = TextEmbedder()
        train_text_emb = embedder.generate_embeddings(df_train)

    # Prepare numeric features
    train_numeric = df_train[Config.NUMERIC_COLS].values.astype(np.float32)

    # Initialize container for OOF predictions
    oof_preds = np.zeros(len(df_train))

    # Re-run inference using saved fold artifacts
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED
    )
    models_dir = os.path.join(Config.WORKING_DIR, "models")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, target)):
        # Load Artifacts
        model_path = os.path.join(models_dir, f"model_fold_{fold}.joblib")
        scaler_path = os.path.join(models_dir, f"scaler_fold_{fold}.joblib")
        encoder_path = os.path.join(models_dir, f"homophily_encoder_fold_{fold}.joblib")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        homophily_encoder = joblib.load(encoder_path)

        # Prepare Validation Data for this fold
        # 1. Text
        fold_text_val = train_text_emb[val_idx]

        # 2. Homophily (Transform validation set using encoder fitted on training set)
        fold_df_val = df_train.iloc[val_idx]
        fold_homo_val = homophily_encoder.transform(fold_df_val)

        # 3. Numeric
        fold_num_val = train_numeric[val_idx]

        # Concatenate
        X_val_raw = np.hstack([fold_text_val, fold_homo_val, fold_num_val])

        # Scale
        X_val = scaler.transform(X_val_raw)

        # Predict
        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs

    # 4. Calculate and Print Metric
    final_auc = roc_auc_score(target, oof_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    # Calculate error magnitude
    errors = np.abs(target - oof_preds)

    # Create analysis dataframe
    df_analysis = df_train[Config.NUMERIC_COLS].copy()
    df_analysis["error"] = errors

    # Calculate correlations
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False, key=abs)
    )

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    print(correlations.head(5).to_string())

    # 6. Submission Logic
    # The Trainer class automatically generates the submission file at Config.SUBMISSION_PATH.
    # We must strictly adhere to the requirement: "Generate predictions... If and only if... > 0.7141749705260098"
    threshold = 0.7141749705260098

    if final_auc <= threshold:
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)
            # print("Submission file removed due to low validation score.") # Suppressed as per instructions
    else:
        # Ensure the file exists (Trainer should have created it)
        if not os.path.exists(Config.SUBMISSION_PATH):
            # Fallback: manually save if Trainer failed to save but score is good (unlikely)
            # We would need to reconstruct test preds here, but Trainer logic is robust.
            pass


if __name__ == "__main__":
    main()
