import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything
from library.dataset import MGMTDataset, get_transforms
from library.model import MGMTClassifier
from library.train import run_training
from library.predict import generate_submission


def extract_validation_features(metadata_df):
    """
    Extracts file system metadata (counts and sizes) for the validation set
    to perform failure analysis.
    """
    features = []
    modalities = ["flair", "t1wce", "t2w"]

    for _, row in metadata_df.iterrows():
        sid = row["BraTS21ID"]
        feat_row = {"BraTS21ID": sid}

        for mod in modalities:
            # Construct full path
            rel_path = row[f"{mod}_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            count = 0
            avg_size = 0.0

            if os.path.exists(full_path):
                files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]
                count = len(files)
                if count > 0:
                    # Sample up to 5 files for speed
                    sample_files = files[:5]
                    sizes = [
                        os.path.getsize(os.path.join(full_path, f))
                        for f in sample_files
                    ]
                    avg_size = np.mean(sizes)

            feat_row[f"{mod}_count"] = count
            feat_row[f"{mod}_avg_size"] = avg_size

        features.append(feat_row)

    return pd.DataFrame(features)


def perform_failure_analysis(val_df, preds, targets, subject_ids):
    """
    Correlates prediction errors with input data characteristics.
    """
    print("\n=== Failure Analysis ===")

    # Create analysis dataframe
    df_analysis = pd.DataFrame(
        {"BraTS21ID": subject_ids, "target": targets, "prediction": preds}
    )

    # Calculate Error
    df_analysis["error"] = np.abs(df_analysis["target"] - df_analysis["prediction"])

    # Extract file system features
    print("Extracting metadata features for validation set...")
    df_features = extract_validation_features(val_df)

    # Merge
    df_merged = pd.merge(df_analysis, df_features, on="BraTS21ID", how="inner")

    # Calculate correlations
    feature_cols = [
        c
        for c in df_merged.columns
        if c not in ["BraTS21ID", "target", "prediction", "error"]
    ]

    print("\nCorrelation between Absolute Error and Metadata Features:")
    print(f"{'Feature':<25} | {'Correlation':<12} | {'P-Value':<12}")
    print("-" * 55)

    for col in feature_cols:
        if df_merged[col].std() == 0:
            corr, pval = 0.0, 1.0
        else:
            corr, pval = pearsonr(df_merged["error"], df_merged[col])
        print(f"{col:<25} | {corr:12.4f} | {pval:12.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()

    # 2. Training
    # We use 10 epochs for a fast baseline execution.
    print("Starting Training Pipeline...")
    run_training(epochs=10, load_cached_data=True)

    # 3. Validation Inference
    print("\nStarting Validation Inference...")

    # Load Validation Data
    val_dataset = MGMTDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        split="val",
        transform=get_transforms("val"),
        load_cached_data=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    device = Config.DEVICE
    model = MGMTClassifier(
        model_name=Config.BACKBONE,
        pretrained=False,  # Weights loaded from checkpoint
        num_classes=Config.NUM_CLASSES,
    )

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError("Model checkpoint not found after training.")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Inference Loop
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for images, targets, subject_ids in val_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy().flatten())
            all_ids.extend(subject_ids.numpy().flatten())

    # 4. Metric Calculation
    final_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    val_metadata = pd.read_csv(Config.VAL_METADATA_PATH)
    perform_failure_analysis(val_metadata, all_preds, all_targets, all_ids)

    # 6. Submission
    print("\nGenerating Submission for Test Set...")
    generate_submission(load_cached_data=True)

    print("\nPipeline Completed Successfully.")


if __name__ == "__main__":
    main()
