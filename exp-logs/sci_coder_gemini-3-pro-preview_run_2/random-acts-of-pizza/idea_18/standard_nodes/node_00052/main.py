import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import setup_logger, set_seed
from library.trainer import run_training_pipeline
from library.data_loader import load_data
from library.embeddings import get_text_embeddings
from library.features import UserPersonaTransformer, MetadataTransformer


def perform_failure_analysis(logger):
    """
    Reconstructs the cross-validation predictions (OOF) to perform failure analysis.
    Since the training pipeline saves models but not the fitted feature transformers,
    this function replicates the feature engineering steps to ensure exact consistency.
    """
    logger.info("Reconstructing OOF predictions for Failure Analysis...")

    # 1. Load Data (Cached)
    df_train, df_val, df_test = load_data(load_cached_data=True)
    emb_train, emb_val, emb_test = get_text_embeddings(
        df_train, df_val, df_test, load_cached_data=True
    )

    # 2. Merge Train and Validation to form Development Set (matching trainer.py logic)
    df_dev = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    emb_dev = np.vstack([emb_train, emb_val])
    y_dev = df_dev["requester_received_pizza"].astype(int).values

    # 3. Initialize CV
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    oof_preds = np.zeros(len(df_dev))

    # 4. Iterate Folds to reproduce predictions
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_dev, y_dev)):
        # Split Data
        X_train_df = df_dev.iloc[train_idx].copy()
        X_val_df = df_dev.iloc[val_idx].copy()
        X_val_emb = emb_dev[val_idx]

        # Fit Transformers (Replicating trainer.py)
        # View 2: User Persona
        persona_transformer = UserPersonaTransformer(
            subreddit_col="subreddit_text",
            n_components=Config.LSA_N_COMPONENTS,
            random_state=Config.SEED,
        )
        persona_transformer.fit(X_train_df)
        X_val_persona = persona_transformer.transform(X_val_df)

        # View 3: Metadata
        meta_transformer = MetadataTransformer(
            numerical_cols=Config.NUMERICAL_COLS, random_state=Config.SEED
        )
        meta_transformer.fit(X_train_df)
        X_val_meta = meta_transformer.transform(X_val_df)

        # Fuse Features: [SBERT, Persona, Metadata]
        X_val_fused = np.hstack([X_val_emb, X_val_persona, X_val_meta])

        # Load Trained Model
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.joblib")
        if not os.path.exists(model_path):
            logger.warning(f"Model for fold {fold} not found. Skipping.")
            continue

        model = joblib.load(model_path)

        # Predict
        val_probs = model.predict_proba(X_val_fused)[:, 1]
        oof_preds[val_idx] = val_probs

    # 5. Calculate Correlations
    # Error Magnitude = |True Label - Predicted Probability|
    errors = np.abs(y_dev - oof_preds)

    # Create analysis dataframe with numerical features
    analysis_df = df_dev[Config.NUMERICAL_COLS].copy()

    # Add text length features for analysis
    analysis_df["text_len_char"] = df_dev["full_text"].fillna("").astype(str).apply(len)
    analysis_df["text_len_word"] = (
        df_dev["full_text"].fillna("").astype(str).apply(lambda x: len(x.split()))
    )

    # Add error column
    analysis_df["error_magnitude"] = errors

    # Compute correlation
    correlations = (
        analysis_df.corr()["error_magnitude"]
        .drop("error_magnitude")
        .sort_values(ascending=False, key=abs)
    )

    print("\n" + "=" * 50)
    print("FAILURE ANALYSIS: Correlation with Error Magnitude")
    print("=" * 50)
    print(correlations)
    print("=" * 50 + "\n")


def main():
    # 1. Setup
    logger = setup_logger(name="runfile")
    set_seed(Config.SEED)

    # 2. Execute Training Pipeline
    # This handles data loading, training, and submission file generation
    logger.info("Starting LPADF Training Pipeline...")
    final_auc = run_training_pipeline(load_cached_data=True)

    # 3. Print Final Metric (Required Format)
    print(f"Final Validation Metric: {final_auc}")

    # 4. Submission Validation
    # Ensure submission exists only if metric threshold is met
    threshold = 0.7141749705260098
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    if final_auc > threshold:
        logger.info(
            f"Validation metric ({final_auc}) exceeds threshold. Submission retained."
        )
    else:
        logger.warning(
            f"Validation metric ({final_auc}) is below threshold ({threshold})."
        )
        if os.path.exists(submission_path):
            logger.info("Removing submission file.")
            os.remove(submission_path)

    # 5. Failure Analysis
    perform_failure_analysis(logger)


if __name__ == "__main__":
    main()
