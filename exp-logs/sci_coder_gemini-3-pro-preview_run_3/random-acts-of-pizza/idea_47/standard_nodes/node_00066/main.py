import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from library.config import Config, set_seed
from library.training_pipeline import CVEnsembleTrainer
from library.utils import setup_logging, get_logger


def main():
    # 1. Setup
    setup_logging()
    logger = get_logger("RunFile")
    set_seed(Config.SEED)

    logger.info("Starting execution of Restored-History Hex-View Stacking Ensemble...")

    # 2. Initialize and Train
    # The CVEnsembleTrainer handles the FeaturePipeline, 5-Fold CV, and Model Persistence
    trainer = CVEnsembleTrainer()

    # Run the training loop to generate OOF predictions and save fold-models
    trainer.train_loop()

    # Train the Meta-Learner on the OOF predictions
    trainer.train_meta_learner()

    # 3. Validation on Hold-out Set
    # We strictly evaluate on the 'val.parquet' subset defined in metadata.
    # Since CV-Bagging merges train and val, we use the OOF predictions for the
    # validation indices to ensure an unbiased metric (no leakage).
    logger.info("Performing validation on hold-out dataset...")

    # Load metadata to determine the exact split indices
    df_train_meta = pd.read_parquet(Config.TRAIN_METADATA_PATH)
    df_val_meta = pd.read_parquet(Config.VAL_METADATA_PATH)

    n_train_samples = len(df_train_meta)
    n_val_samples = len(df_val_meta)

    # Load the OOF predictions generated during training
    oof_path = os.path.join(trainer.predictions_dir, "oof_predictions.parquet")
    if not os.path.exists(oof_path):
        raise FileNotFoundError("OOF predictions not found. Training may have failed.")

    oof_preds_df = pd.read_parquet(oof_path)

    # The pipeline concatenates [train_part, val_part].
    # We slice the tail to get the validation set OOF predictions.
    val_oof_preds = oof_preds_df.iloc[n_train_samples : n_train_samples + n_val_samples]

    # Get the corresponding ground truth
    # trainer.y_train contains the concatenated targets
    y_val_true = trainer.y_train[n_train_samples : n_train_samples + n_val_samples]

    # Load the trained Meta-Learner
    meta_model_path = os.path.join(trainer.models_dir, "meta_learner.joblib")
    meta_model = joblib.load(meta_model_path)

    # Generate final probabilities for the validation set using the Meta-Learner
    X_meta_val = val_oof_preds[trainer.model_names].values
    y_val_pred = meta_model.predict_proba(X_meta_val)[:, 1]

    # Compute and print the required metric
    val_auc = roc_auc_score(y_val_true, y_val_pred)
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    logger.info("Performing Failure Analysis on Validation Set...")

    # Calculate absolute error
    errors = np.abs(y_val_true - y_val_pred)

    # Retrieve contextual features for the validation subset
    # trainer.X_train_dict['contextual'] is the scaled dense metadata
    X_val_ctx = trainer.X_train_dict["contextual"][
        n_train_samples : n_train_samples + n_val_samples
    ]

    # Calculate correlation between each feature and the error
    feature_names = Config.DENSE_FEATURES
    correlations = []

    # Ensure dimensions match (in case feature list changed, though unlikely)
    n_features = min(len(feature_names), X_val_ctx.shape[1])

    for i in range(n_features):
        feat_vals = X_val_ctx[:, i]
        # Handle constant features to avoid NaN correlation
        if np.std(feat_vals) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_vals, errors)[0, 1]
        correlations.append((feature_names[i], corr))

    # Sort by correlation magnitude (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Failure Analysis - Top Feature Correlations with Model Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 5. Submission Generation
    # Condition: Generate submission only if validation score > threshold
    THRESHOLD = 0.7138293787137718

    if val_auc > THRESHOLD:
        logger.info(
            f"Validation AUC ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission()
    else:
        logger.warning(
            f"Validation AUC ({val_auc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
