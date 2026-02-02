import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config, set_seed
from library.utils import get_logger, get_device
from library.train import run_training, generate_submission
from library.data_loader import get_dataloaders
from library.model import MGMT25DModel


def main():
    # 1. Setup & Configuration
    logger = get_logger()
    set_seed(Config.SEED)
    device = get_device()

    logger.info("Starting Runfile Execution...")

    # Override Config for Fast Baseline as per requirements
    # Reducing epochs to 10 ensures quick execution while allowing sufficient convergence
    Config.EPOCHS = 10

    # 2. Training Phase
    # run_training handles the full training loop, early stopping, and saving the best model.
    # It returns the path to the best saved model and the test_loader for submission.
    best_model_path, test_loader = run_training()

    # 3. Validation & Failure Analysis Phase
    logger.info("Starting Validation and Failure Analysis...")

    # Retrieve the validation loader. We use load_cached_data=True to speed up loading.
    # run_training has likely already cached the data.
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Load the best model for evaluation
    model = MGMT25DModel().to(device)
    if not os.path.exists(best_model_path):
        logger.error(f"Best model not found at {best_model_path}. Exiting.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Run Inference on Validation Set
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            val_preds.extend(probs)
            val_targets.extend(targets.cpu().numpy().flatten())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate and Print Final Metric
    final_auc = roc_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlate Error with Metadata Features
    # Retrieve IDs from the dataset to link predictions with metadata
    val_ids = val_loader.dataset.ids
    errors = np.abs(val_targets - val_preds)

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame(
        {
            "BraTS21ID": val_ids,
            "error": errors,
            "target": val_targets,
            "prediction": val_preds,
        }
    )

    # Load validation metadata to extract features (e.g., slice counts)
    val_meta_df = pd.read_parquet(Config.VAL_META_PATH)

    # Merge analysis results with metadata
    merged_df = pd.merge(val_meta_df, analysis_df, on="BraTS21ID")

    # Feature Engineering: Calculate Total Slices per patient
    def get_total_slices(row):
        count = 0
        for mod in ["flair", "t1w", "t1wce", "t2w"]:
            paths = row.get(f"{mod}_paths")
            if isinstance(paths, (list, np.ndarray)):
                count += len(paths)
        return count

    merged_df["total_slices"] = merged_df.apply(get_total_slices, axis=1)

    # Calculate Correlations
    logger.info("Performing failure analysis correlations...")

    if len(merged_df) > 1:
        # Correlation between Error and Total Slice Count
        # High correlation suggests the model struggles with varying volume depths
        corr_slices, _ = pearsonr(merged_df["error"], merged_df["total_slices"])
        print(f"Correlation (Error vs Total Slices): {corr_slices}")

        # Correlation between Error and Target Class
        # High correlation suggests bias towards one class
        corr_target, _ = pearsonr(merged_df["error"], merged_df["target"])
        print(f"Correlation (Error vs Target Class): {corr_target}")

    # 4. Submission Phase
    # Threshold required by the task
    THRESHOLD = 0.6978181818181817

    if final_auc > THRESHOLD:
        logger.info(
            f"Validation AUC ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(best_model_path, test_loader)
    else:
        logger.info(
            f"Validation AUC ({final_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
