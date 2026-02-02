import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.trainer import Trainer
from library.dataset import get_datasets
from library.data_processing import get_roi_cache


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    logger = get_logger("Runfile")

    # 2. Training
    # The dataset is small (~500 samples), so the default configuration (10 epochs)
    # serves as a fast baseline while ensuring sufficient convergence.
    logger.info("Initializing Trainer...")
    trainer = Trainer()

    logger.info("Starting Training Loop...")
    # fit() handles training, validation monitoring, and checkpointing
    trainer.fit(load_cached_data=True)

    # 3. Final Validation Assessment
    logger.info("Running Final Validation Assessment...")

    # Load the best model saved during training
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        logger.error("Model checkpoint not found. Training might have failed.")
        return

    trainer.model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=trainer.device)
    )
    trainer.model.eval()

    # Get validation dataset
    # We use get_datasets to ensure consistent caching/loading logic
    _, val_dataset, _ = get_datasets(load_cached_data=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(trainer.device.type == "cuda"),
    )

    all_targets = []
    all_probs = []
    all_ids = []

    # Optimized inference with no gradients
    with torch.no_grad():
        for inputs, labels, subject_ids in val_loader:
            inputs = inputs.to(trainer.device)

            # Forward pass
            outputs = trainer.model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_targets.extend(labels.numpy().flatten())
            all_ids.extend(subject_ids)

    # Compute Metric
    # Handle edge case where batch might have only one class (unlikely for full val set)
    if len(np.unique(all_targets)) > 1:
        final_metric = roc_auc_score(all_targets, all_probs)
    else:
        final_metric = 0.5

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(
        {"BraTS21ID": all_ids, "target": all_targets, "prediction": all_probs}
    )

    # Calculate Error Magnitude
    df_analysis["error"] = np.abs(df_analysis["prediction"] - df_analysis["target"])

    # Extract Metadata Features for Correlation
    # 1. ROI Index (from cache)
    roi_cache = val_dataset.roi_cache
    df_analysis["roi_index"] = df_analysis["BraTS21ID"].map(roi_cache)

    # 2. Slice Counts (Proxy for brain volume/scan resolution)
    # Map ID to FLAIR path to check file counts
    path_map = pd.Series(
        val_dataset.df.path_FLAIR.values, index=val_dataset.df.BraTS21ID.astype(str)
    ).to_dict()

    def get_slice_count(subject_id):
        path = path_map.get(subject_id)
        if path:
            full_path = os.path.join(Config.INPUT_DIR, path)
            if os.path.exists(full_path):
                # Fast directory listing
                return len(
                    [
                        name
                        for name in os.listdir(full_path)
                        if os.path.isfile(os.path.join(full_path, name))
                    ]
                )
        return 0

    df_analysis["flair_slices"] = df_analysis["BraTS21ID"].apply(get_slice_count)

    # Compute Correlations
    features_to_analyze = ["roi_index", "flair_slices", "target"]
    print("\nCorrelation between Error Magnitude and Input Features:")
    for feat in features_to_analyze:
        if feat in df_analysis.columns:
            # Handle potential NaNs or constant values
            if df_analysis[feat].std() > 0:
                corr = df_analysis["error"].corr(df_analysis[feat])
                print(f"Feature: {feat}, Correlation: {corr}")
            else:
                print(f"Feature: {feat}, Correlation: NaN (Constant value)")

    # 5. Submission Generation
    # Threshold check
    SUBMISSION_THRESHOLD = 0.6254545454545455

    if final_metric > SUBMISSION_THRESHOLD:
        logger.info(
            f"Validation metric ({final_metric}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        trainer.predict(load_cached_data=True)
    else:
        logger.warning(
            f"Validation metric ({final_metric}) did not exceed threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
