import sys
import os
import pandas as pd
import numpy as np
import torch

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, compute_roc_auc
from library.data_loader import get_dataloaders
from library.trainer import Trainer
from library.model import S3HDNetwork


def main():
    # 1. Setup Environment
    # Initialize configuration (creates directories, sets paths)
    Config.setup()

    # Set random seeds for reproducibility
    set_seed(Config.SEED)

    print("Environment initialized.")

    # 2. Data Loading
    # Load DataLoaders with caching enabled to speed up subsequent runs
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)
    print("Data loaded successfully.")

    # 3. Model Training
    # Initialize the Trainer which manages the model, optimizer, and training loop
    trainer = Trainer()

    print("Starting training...")
    # Train the model with early stopping and checkpointing
    trainer.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS)
    print("Training complete.")

    # 4. Validation Assessment
    print("Performing final validation assessment...")

    # Load the best model saved during training
    device = trainer.device
    model = trainer.model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model weights.")

    model.eval()

    val_targets = []
    val_probs = []
    val_ids = []

    # Run inference on validation set
    with torch.no_grad():
        for inputs, targets, batch_ids in val_loader:
            inputs = inputs.to(device)
            # Forward pass
            logits = model(inputs)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            val_targets.extend(targets.numpy())
            val_probs.extend(probs)
            val_ids.extend(batch_ids)

    # Compute Final Metric
    final_metric = compute_roc_auc(val_targets, val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")

    # Create analysis DataFrame
    analysis_df = pd.DataFrame(
        {"BraTS21ID": val_ids, "target": val_targets, "prob": val_probs}
    )

    # Calculate absolute error
    analysis_df["error"] = np.abs(analysis_df["target"] - analysis_df["prob"])

    # Load metadata to retrieve features (slice counts)
    val_meta_df = pd.read_parquet(Config.VAL_META_PATH)

    # Extract meta-features for correlation analysis
    # We calculate slice counts as a proxy for information density
    meta_features = []
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    for idx, row in val_meta_df.iterrows():
        feat = {"BraTS21ID": row["BraTS21ID"]}
        for mod in modalities:
            col_name = f"{mod}_paths"
            paths = row[col_name] if row[col_name] is not None else []
            feat[f"{mod}_count"] = len(paths)
        meta_features.append(feat)

    meta_feat_df = pd.DataFrame(meta_features)

    # Merge error data with meta-features
    # Ensure ID types match (both string)
    analysis_df["BraTS21ID"] = analysis_df["BraTS21ID"].astype(str)
    meta_feat_df["BraTS21ID"] = meta_feat_df["BraTS21ID"].astype(str)

    full_analysis_df = pd.merge(analysis_df, meta_feat_df, on="BraTS21ID", how="left")

    # Calculate and print correlations
    print("Correlation between Model Error and Input Features:")
    feature_cols = [f"{mod}_count" for mod in modalities]

    for col in feature_cols:
        if col in full_analysis_df.columns:
            if full_analysis_df[col].std() > 0:
                corr = full_analysis_df["error"].corr(full_analysis_df[col])
                print(f"  {col}: {corr}")
            else:
                print(f"  {col}: NaN (No variance)")

    # 6. Submission Generation
    # Threshold defined in requirements
    THRESHOLD = 0.6978181818181817

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        # Trainer.predict handles loading the best model and saving the CSV
        trainer.predict(test_loader)
    else:
        print(
            f"\nValidation metric ({final_metric}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
