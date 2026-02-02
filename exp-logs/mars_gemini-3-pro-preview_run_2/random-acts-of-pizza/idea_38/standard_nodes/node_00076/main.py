import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import load_and_process_data
from library.trainer import Trainer


def main():
    # ---------------------------------------------------------
    # 1. Setup & Initialization
    # ---------------------------------------------------------
    logger = setup_logger("runfile")
    set_seed(Config.SEED)

    logger.info("Initializing Trainer...")
    trainer = Trainer()

    # ---------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------
    # We use the full dataset (debug=False) as the dataset size (~3k rows)
    # allows for fast training within the 2-hour limit.
    logger.info("Starting Training process...")
    trainer.train(debug=False, load_cached_data=True)

    # ---------------------------------------------------------
    # 3. Validation Assessment
    # ---------------------------------------------------------
    logger.info("Starting Validation Assessment...")

    # Load the specific validation split
    # load_and_process_data returns (train, val, test)
    _, df_val, _ = load_and_process_data(debug=False, load_cached_data=True)

    # Retrieve embeddings for the validation set using the trainer's embedding service
    # This ensures we use the same model/caching logic
    anchor_val = trainer.embedding_service.get_embeddings(
        df_val, "val", "anchor", load_cached_data=True
    )
    aux_val = trainer.embedding_service.get_embeddings(
        df_val, "val", "aux", load_cached_data=True
    )

    # Construct the feature matrix for validation
    # We must align the DataFrame index with the numpy arrays
    df_val = df_val.reset_index(drop=True)

    # Create column names matching the training pipeline expectation
    anchor_cols = [f"anchor_{i}" for i in range(anchor_val.shape[1])]
    aux_cols = [f"aux_{i}" for i in range(aux_val.shape[1])]

    df_anchor = pd.DataFrame(anchor_val, columns=anchor_cols, index=df_val.index)
    df_aux = pd.DataFrame(aux_val, columns=aux_cols, index=df_val.index)

    # Concatenate: Metadata + Anchor Embeddings + Aux Embeddings
    X_val = pd.concat([df_val, df_anchor, df_aux], axis=1)
    y_val = df_val[Config.TARGET_COL].values

    # Load all fold models and predict
    fold_preds = []
    models_found = 0

    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.joblib")
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                # Predict probability of class 1
                preds = model.predict_proba(X_val)[:, 1]
                fold_preds.append(preds)
                models_found += 1
            except Exception as e:
                logger.error(f"Error loading model fold {fold}: {e}")
        else:
            logger.warning(f"Model for fold {fold} not found.")

    if models_found == 0:
        logger.error("No trained models found. Cannot perform validation.")
        return

    # Average predictions (Ensemble)
    avg_preds = np.mean(fold_preds, axis=0)

    # Compute and print the Final Validation Metric
    val_score = roc_auc_score(y_val, avg_preds)
    print(f"Final Validation Metric: {val_score}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    logger.info("Performing Failure Analysis on Validation Set...")

    # Calculate absolute error
    errors = np.abs(y_val - avg_preds)

    # Create a temporary dataframe for correlation analysis
    # We analyze numerical features defined in Config
    analysis_df = df_val[Config.NUMERICAL_COLS].copy()
    analysis_df["error"] = errors

    # Compute correlation between features and error
    correlations = analysis_df.corr()["error"].drop("error")

    # Sort by absolute correlation
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)

    logger.info("Top 5 Features correlated with Prediction Error:")
    logger.info(f"\n{top_correlations}")

    # ---------------------------------------------------------
    # 5. Submission Generation
    # ---------------------------------------------------------
    threshold = 0.7190361601447052

    if val_score > threshold:
        logger.info(
            f"Validation score ({val_score}) exceeds threshold ({threshold}). Generating submission..."
        )
        trainer.generate_submission(debug=False, load_cached_data=True)
    else:
        logger.warning(
            f"Validation score ({val_score}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
