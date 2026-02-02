import sys
import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add current directory to path to ensure local library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger, WGS84
from library.data_processing import GNSSPreprocessor
from library.dataset import SKFDataset
from library.trainer import Trainer
from library.inference import InferenceEngine


def calculate_competition_metric(df_results):
    """
    Calculates the competition metric:
    Mean of the (50th percentile + 95th percentile) / 2 per phone (trip).
    """

    def get_trip_score(group):
        errors = group["error_meters"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        return (p50 + p95) / 2

    trip_scores = df_results.groupby("tripId").apply(get_trip_score)
    return trip_scores.mean()


def main():
    # 1. Setup
    warnings.filterwarnings("ignore")
    seed_everything(Config.RANDOM_SEED)
    logger = get_logger("runfile")

    logger.info("Starting SKF-Net Pipeline...")

    # 2. Data Preparation
    preprocessor = GNSSPreprocessor()

    logger.info("Loading and processing Training Data...")
    # load_cached_data=True will use existing .npy files if available in working dir
    X_train_seq, X_train_sky, y_train, train_meta = preprocessor.process_train(
        load_cached_data=True
    )

    logger.info("Loading and processing Validation Data...")
    X_val_seq, X_val_sky, y_val, val_meta = preprocessor.process_val(
        load_cached_data=True
    )

    # Sanity check
    if len(X_train_seq) == 0 or len(X_val_seq) == 0:
        logger.error("Data loading failed. Arrays are empty.")
        return

    # Create PyTorch Datasets
    train_dataset = SKFDataset(X_train_seq, X_train_sky, y_train)
    val_dataset = SKFDataset(X_val_seq, X_val_sky, y_val)

    # 3. Training
    logger.info("Initializing Trainer...")
    trainer = Trainer()

    logger.info("Starting Training Loop...")
    # This saves the best model to Config.MODEL_PATH
    trainer.fit(train_dataset, val_dataset)

    # 4. Validation Assessment
    logger.info("Running Validation Inference for Metric Calculation...")

    # We need to run inference manually here to get the residuals for metric calc
    # The trainer.validate() only returns loss (MAE), not individual predictions
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    trainer.model.eval()

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for x_seq, x_sky, y in val_loader:
            x_seq = x_seq.to(trainer.device)
            x_sky = x_sky.to(trainer.device)

            # Predict residuals (meters)
            output = trainer.model(x_seq, x_sky)

            preds_list.append(output.cpu().numpy())
            targets_list.append(y.cpu().numpy())

    preds_residuals = np.concatenate(preds_list, axis=0)
    target_residuals = np.concatenate(targets_list, axis=0)

    # Calculate Euclidean distance error in meters
    # The model predicts [DeltaEast, DeltaNorth]
    # The target is [DeltaEast, DeltaNorth]
    # Error is the magnitude of the difference vector
    diff = preds_residuals - target_residuals
    error_meters = np.sqrt(np.sum(diff**2, axis=1))

    # Create results DataFrame
    val_results = pd.DataFrame(
        {"tripId": val_meta["tripId"], "error_meters": error_meters}
    )

    # Compute Metric
    final_metric = calculate_competition_metric(val_results)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # We correlate the error magnitude with the Sky Context features
    # X_val_sky contains the scaled features. Correlations are invariant to linear scaling.
    feature_names = Config.SKY_FEATURES

    print("-" * 30)
    print("Correlation between Error and Sky Features:")
    print("-" * 30)

    correlations = {}
    for i, name in enumerate(feature_names):
        # Extract feature column
        feat_values = X_val_sky[:, i]
        # Calculate Pearson correlation
        corr = np.corrcoef(feat_values, error_meters)[0, 1]
        correlations[name] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(
        correlations.items(), key=lambda item: abs(item[1]), reverse=True
    )

    for name, corr in sorted_corrs:
        print(f"{name}: {corr:.4f}")
    print("-" * 30)

    # 6. Submission Generation
    THRESHOLD = 4.256982128481356

    if final_metric < THRESHOLD:
        logger.info(
            f"Validation metric ({final_metric:.4f}) is better than threshold ({THRESHOLD:.4f})."
        )
        logger.info("Generating submission file...")

        inference_engine = InferenceEngine()
        # This processes test data, loads the best model, and creates submission.csv
        inference_engine.generate_submission(load_cached_data=True)

    else:
        logger.warning(
            f"Validation metric ({final_metric:.4f}) did not meet threshold ({THRESHOLD:.4f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
