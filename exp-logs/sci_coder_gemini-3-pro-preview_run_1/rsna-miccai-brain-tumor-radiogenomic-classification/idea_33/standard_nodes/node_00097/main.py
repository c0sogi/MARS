import os
import sys
import torch
import pandas as pd
import numpy as np
import logging

# Import from the provided library files
from library.config import WORKING_DIR, DEVICE, SEED, VAL_METADATA_PATH, INPUT_DIR
from library.utils import (
    seed_everything,
    get_logger,
    load_checkpoint,
    calculate_roc_auc,
)
from library.model import RARVEfficientNet
from library.trainer import run_training
from library.data_processing import get_dataloaders


def main():
    # 1. Setup and Configuration
    # Ensure reproducibility
    seed_everything(SEED)

    # Create submission directory
    os.makedirs("./submission", exist_ok=True)

    # Initialize logger
    logger = get_logger("runfile")
    logger.info("Initializing RARV Pipeline Execution...")

    # 2. Model Training
    # We use the trainer from library/trainer.py.
    # We limit epochs to 10 to ensure the baseline runs quickly within the time limit.
    # The trainer handles data loading, model initialization, and the training loop.
    logger.info("Starting Training Phase...")
    run_training(load_cached_data=True, max_epochs=15)

    # 3. Model Loading for Inference
    # Load the best model saved during training
    logger.info("Loading best model for evaluation...")
    model = RARVEfficientNet().to(DEVICE)

    # load_checkpoint looks in WORKING_DIR for the filename
    try:
        load_checkpoint(model, filename="best_model.pth", device=DEVICE)
    except FileNotFoundError:
        logger.error("Best model checkpoint not found. Training may have failed.")
        return

    model.eval()

    # 4. Validation Inference
    # Get dataloaders (re-using cached data)
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    val_probs = []
    val_targets = []
    val_ids = []

    logger.info("Running Inference on Validation Set...")

    with torch.no_grad():
        for images, targets, subject_ids in val_loader:
            images = images.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            val_probs.extend(probs)
            val_targets.extend(targets.numpy().flatten())
            val_ids.extend(subject_ids)

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)

    # 5. Metric Calculation
    # Calculate and print the final validation metric as required
    final_metric = calculate_roc_auc(val_targets, val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis on Validation Set...")

    # Load validation metadata to retrieve file paths for analysis
    df_val_meta = pd.read_csv(VAL_METADATA_PATH)

    # Create an analysis dataframe
    # Ensure IDs are integers for merging
    val_ids_int = [int(x) for x in val_ids]

    df_analysis = pd.DataFrame(
        {
            "BraTS21ID": val_ids_int,
            "target": val_targets,
            "prediction": val_probs,
            "error": np.abs(val_targets - val_probs),
        }
    )

    # Merge with metadata
    df_analysis = df_analysis.merge(df_val_meta, on="BraTS21ID", how="left")

    # Feature Extraction: Count FLAIR slices as a proxy for brain volume/scan depth
    # This checks if the model struggles with smaller or larger volumes
    flair_counts = []
    for _, row in df_analysis.iterrows():
        try:
            # Construct full path
            flair_path = os.path.join(INPUT_DIR, row["flair_path"])
            if os.path.exists(flair_path):
                # Count dicom files
                count = len([f for f in os.listdir(flair_path) if f.endswith(".dcm")])
                flair_counts.append(count)
            else:
                flair_counts.append(0)
        except Exception:
            flair_counts.append(0)

    df_analysis["flair_slice_count"] = flair_counts

    # Calculate Correlation
    if len(df_analysis) > 1:
        corr_depth = df_analysis["error"].corr(df_analysis["flair_slice_count"])
        print(f"Correlation between Error and FLAIR Slice Count: {corr_depth:.8f}")

        # Additional check: Correlation with Target class (is it harder to predict 0 or 1?)
        corr_class = df_analysis["error"].corr(df_analysis["target"])
        print(f"Correlation between Error and Target Class: {corr_class:.8f}")

    # 7. Submission Generation
    # Threshold defined in the task
    THRESHOLD = 0.6705454545454544

    if final_metric > THRESHOLD:
        logger.info(
            f"Validation metric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        test_probs = []
        test_ids = []

        with torch.no_grad():
            for images, _, subject_ids in test_loader:
                images = images.to(DEVICE)

                # Forward pass
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                test_probs.extend(probs)
                test_ids.extend(subject_ids)

        # Create submission DataFrame
        df_submission = pd.DataFrame(
            {"BraTS21ID": [int(x) for x in test_ids], "MGMT_value": test_probs}
        )

        # Sort by ID (good practice)
        df_submission = df_submission.sort_values("BraTS21ID")

        # Save to file
        submission_path = "./submission/submission.csv"
        df_submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")

    else:
        logger.warning(
            f"Validation metric ({final_metric}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
