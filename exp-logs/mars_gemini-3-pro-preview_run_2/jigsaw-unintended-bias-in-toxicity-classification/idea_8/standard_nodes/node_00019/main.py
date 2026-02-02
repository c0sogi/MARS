import os
import sys
import numpy as np
import pandas as pd
import torch
import gc
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.data_processing import (
    get_dataloaders,
    get_test_dataloader,
    ToxicityDataset,
)
from library.trainer import Trainer
from library.model import ToxicityModel
from library.utils import seed_everything, get_device
from library.metrics import compute_bias_metrics


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Baseline
    # --------------------------------------------------------------------------
    # Adjust configuration to ensure execution within 2 hours
    print("Applying configuration overrides for fast execution...")
    Config.EPOCHS = 1
    Config.USE_SWA = False
    Config.MAX_LEN = 320  # Reduced from 512 to speed up tokenization and training

    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = get_device()

    print(f"Device: {device}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Max Length: {Config.MAX_LEN}")

    # --------------------------------------------------------------------------
    # 2. Data Preparation
    # --------------------------------------------------------------------------
    print("\n=== Loading Data ===")
    # load_cached_data=True will look for files in working/idea_8 (Config.WORKING_DIR)
    train_loader, val_loader = get_dataloaders(load_cached_data=True)

    # --------------------------------------------------------------------------
    # 3. Training
    # --------------------------------------------------------------------------
    print("\n=== Starting Training ===")
    trainer = Trainer(train_loader, val_loader)
    trainer.train()

    # Retrieve best score from trainer
    best_score = trainer.best_score
    # REQUIRED: Print the final validation metric in the exact requested format
    print(f"Final Validation Metric: {best_score}")

    # Free up memory from training components
    del trainer.optimizer, trainer.scaler, trainer.scheduler
    if hasattr(trainer, "swa_model"):
        del trainer.swa_model
    torch.cuda.empty_cache()
    gc.collect()

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n=== Performing Failure Analysis ===")

    # Load validation metadata to get all features (identities + subtypes)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # We need to generate predictions for the validation set again to map them to metadata
    # (Trainer.validate computes metrics but doesn't return the full array aligned with metadata df)

    print("Generating validation predictions for analysis...")
    # Load best model
    model = ToxicityModel()
    model.load_state_dict(torch.load(Config.OUTPUT_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    val_preds = []
    val_ids = []

    # Disable gradients for inference
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ids = batch["id"].cpu().numpy()

            # Device-side trimming (same as in Trainer)
            max_len = attention_mask.sum(dim=1).max().item()
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]

            # Forward pass
            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]
            preds = torch.sigmoid(logits).cpu().numpy()

            val_preds.extend(preds)
            val_ids.extend(ids)

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame({Config.ID_COL: val_ids, "prediction": val_preds})

    # Merge with metadata
    analysis_df = val_meta.merge(analysis_df, on=Config.ID_COL, how="inner")

    # Calculate Error: |Target - Prediction|
    analysis_df["error"] = (
        analysis_df[Config.TARGET_COL] - analysis_df["prediction"]
    ).abs()

    # Features to correlate: Toxicity Subtypes + Identity Columns
    features_to_analyze = Config.AUX_COLUMNS + Config.IDENTITY_COLUMNS

    print("\nCorrelation between Error Magnitude and Features:")
    correlations = {}
    for feature in features_to_analyze:
        if feature in analysis_df.columns:
            # Ensure numeric
            if pd.api.types.is_numeric_dtype(analysis_df[feature]):
                corr = analysis_df["error"].corr(analysis_df[feature])
                correlations[feature] = corr

    # Sort and print
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    for feat, corr in sorted_corr:
        print(f"  {feat}: {corr:.4f}")

    # --------------------------------------------------------------------------
    # 5. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9273793163893314

    if best_score > THRESHOLD:
        print(
            f"\nScore ({best_score:.6f}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_loader = get_test_dataloader(load_cached_data=True)

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                ids = batch["id"].cpu().numpy()

                # Device-side trimming
                max_len = attention_mask.sum(dim=1).max().item()
                input_ids = input_ids[:, :max_len]
                attention_mask = attention_mask[:, :max_len]

                outputs = model(input_ids, attention_mask)
                logits = outputs["logits"]
                preds = torch.sigmoid(logits).cpu().numpy()

                test_preds.extend(preds)
                test_ids.extend(ids)

        submission_df = pd.DataFrame(
            {Config.ID_COL: test_ids, "prediction": test_preds}
        )

        # Ensure correct format
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nScore ({best_score:.6f}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
