import os
import sys
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.trainer import Trainer
from library.dataset import get_dataloader
from library.inference import predict_and_submit
from library.utils import seed_everything, log_message


def analyze_failures(trainer, val_loader):
    """
    Performs failure analysis on the validation set.
    Correlates prediction error with input data properties (file counts).
    """
    log_message("Starting Failure Analysis...")

    # 1. Generate Predictions on Validation Set
    trainer.model.eval()
    device = trainer.device

    all_probs = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch (images, labels, ids)
            images, labels, subject_ids = batch

            images = images.to(device)
            outputs = trainer.model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_targets.extend(labels.numpy().flatten())
            all_ids.extend(subject_ids.numpy().flatten())

    # 2. Aggregate to Subject Level (Mean of 3 slabs)
    df_res = pd.DataFrame(
        {"BraTS21ID": all_ids, "prob": all_probs, "target": all_targets}
    )

    # Group by Subject ID
    df_subject = (
        df_res.groupby("BraTS21ID")
        .agg({"prob": "mean", "target": "first"})
        .reset_index()
    )

    # Calculate Absolute Error
    df_subject["error"] = (df_subject["target"] - df_subject["prob"]).abs()

    # 3. Extract Meta-Features (File Counts)
    # We load the validation metadata to get paths
    df_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Merge error data with metadata
    df_analysis = pd.merge(df_subject, df_meta, on="BraTS21ID")

    # Calculate file counts for each modality as a proxy for data quality
    feature_correlations = {}
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    for mod in modalities:
        counts = []
        for _, row in df_analysis.iterrows():
            # Construct full path
            rel_path = row[f"{mod.lower()}_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            # Count DICOM files
            try:
                if os.path.exists(full_path):
                    n_files = len(
                        [f for f in os.listdir(full_path) if f.endswith(".dcm")]
                    )
                else:
                    n_files = 0
            except Exception:
                n_files = 0
            counts.append(n_files)

        col_name = f"{mod}_count"
        df_analysis[col_name] = counts

        # Calculate Correlation with Error
        if len(counts) > 0 and np.std(counts) > 0:
            corr = np.corrcoef(df_analysis["error"], counts)[0, 1]
            feature_correlations[col_name] = corr
        else:
            feature_correlations[col_name] = 0.0

    # 4. Print Results
    print("-" * 30)
    print("Failure Analysis: Correlation between Error and Features")
    print("-" * 30)
    for feature, corr in feature_correlations.items():
        print(f"{feature}: {corr:.6f}")
    print("-" * 30)


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Ensure fast baseline execution
    # The default Config has 10 epochs, which is appropriate for this small dataset.
    # We will ensure DEBUG is False to run on full data (it's small enough).
    Config.DEBUG = False

    # 2. Train
    trainer = Trainer()
    trainer.fit(load_cached_data=True)

    # 3. Validation
    # Load the best model weights
    if os.path.exists(Config.MODEL_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.MODEL_PATH, map_location=trainer.device)
        )

    # Get validation loader
    val_loader = get_dataloader("val", load_cached=True)

    # Compute metric
    _, final_metric = trainer.validate(val_loader)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    analyze_failures(trainer, val_loader)

    # 5. Submission
    THRESHOLD = 0.6705454545454544

    if final_metric > THRESHOLD:
        log_message(
            f"Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(load_cached_data=True)
    else:
        log_message(
            f"Metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
