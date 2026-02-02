import os
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

# Import provided library modules
from library.config import KADM_CONFIG
from library.utils import seed_everything, setup_logger
from library.data_loader import DataLoader
from library.training_curriculum import HardNegativeMiner, ExpertTrainer
from library.inference_engine import ThresholdOptimizer, generate_submission
from library.model_zoo import DualModelEnsemble


def main():
    # 1. Setup and Configuration
    seed_everything(42)
    logger = setup_logger(name="runfile")

    # Override configuration for Fast Baseline Execution
    config = KADM_CONFIG.copy()
    # Reduce boosting rounds for speed
    config["training"]["num_boost_round"] = 300
    config["training"]["early_stopping_rounds"] = 30
    # Ensure parallel processing
    config["settings"]["n_jobs"] = 12
    # Ensure GPU acceleration for XGBoost
    config["models"]["xgb"]["device"] = "cuda"
    config["models"]["xgb"]["tree_method"] = "hist"

    logger.info("Configuration updated for fast baseline execution.")

    # 2. Data Loading
    loader = DataLoader(config)

    # Load Training Data
    logger.info("Loading Training Data...")
    X_train, y_train, _ = loader.load_dataset(
        "train", apply_gating=True, load_cached_data=True
    )

    # Subsample Training Data to ensure the script finishes within the time limit
    # We use 200,000 samples which is sufficient for a strong baseline but fast to train
    MAX_TRAIN_SAMPLES = 200000
    if len(X_train) > MAX_TRAIN_SAMPLES:
        logger.info(
            f"Subsampling training data from {len(X_train)} to {MAX_TRAIN_SAMPLES}..."
        )
        # Use fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.choice(len(X_train), MAX_TRAIN_SAMPLES, replace=False)
        X_train = X_train.iloc[indices].reset_index(drop=True)
        y_train = y_train.iloc[indices].reset_index(drop=True)

    # Load Validation Data (Keep full set for accurate metric calculation)
    logger.info("Loading Validation Data...")
    X_val, y_val, _ = loader.load_dataset(
        "val", apply_gating=True, load_cached_data=True
    )

    # 3. Hard Negative Mining
    logger.info("Starting Hard Negative Mining on subsampled training set...")
    miner = HardNegativeMiner(config)
    # We set load_cached_data=False to force mining on the current subsampled X_train
    # This ensures indices match the current dataframe
    hard_neg_indices = miner.mine(X_train, y_train, load_cached_data=False)

    # 4. Expert Ensemble Training
    logger.info("Training Expert Dual-Model Ensemble...")
    trainer = ExpertTrainer(config)
    ensemble = trainer.fit(X_train, y_train, hard_neg_indices, X_val, y_val)

    # Save the trained ensemble
    ensemble.save()

    # 5. Validation and Threshold Optimization
    logger.info("Optimizing decision threshold on Validation set...")
    optimizer = ThresholdOptimizer(config)
    best_threshold = optimizer.optimize(ensemble, X_val, y_val)

    # Calculate Final Validation Metric on the full validation set
    logger.info("Calculating Final Validation Metric...")
    # Ensure model is in eval mode (implicit in sklearn-style predict)
    preds_prob = ensemble.predict(X_val)
    preds_bin = (preds_prob > best_threshold).astype(int)
    final_mcc = matthews_corrcoef(y_val, preds_bin)

    # Print the required metric string
    print(f"Final Validation Metric: {final_mcc}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate error magnitude (Absolute difference between probability and true label)
    error_magnitude = np.abs(y_val - preds_prob)

    # Select numerical features for correlation analysis
    numeric_features = X_val.select_dtypes(include=[np.number])

    # Compute correlation between features and error magnitude
    correlations = numeric_features.corrwith(
        pd.Series(error_magnitude, index=X_val.index)
    )

    # Get top 10 features most correlated with error
    sorted_corr = correlations.abs().sort_values(ascending=False).head(10)

    print("\nTop 10 Feature Correlations with Error Magnitude:")
    for feature, corr_val in sorted_corr.items():
        # Print original signed correlation to show direction
        original_corr = correlations[feature]
        print(f"{feature}: {original_corr:.6f}")

    # 7. Submission Generation
    TARGET_METRIC = 0.6865
    if final_mcc > TARGET_METRIC:
        logger.info(
            f"Validation MCC ({final_mcc}) exceeds target ({TARGET_METRIC}). Generating submission..."
        )
        generate_submission(
            ensemble=ensemble,
            threshold=best_threshold,
            load_cached_data=True,
            config=config,
        )
    else:
        logger.warning(
            f"Validation MCC ({final_mcc}) did not meet target ({TARGET_METRIC}). Submission skipped."
        )


if __name__ == "__main__":
    main()
