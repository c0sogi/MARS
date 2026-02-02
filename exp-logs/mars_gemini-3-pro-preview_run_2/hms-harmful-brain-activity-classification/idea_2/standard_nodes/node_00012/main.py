import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# Import from provided library files
from library.config import Config
from library.engine import Engine
from library.utils import seed_everything, kl_divergence_score
from library.dataset import EEGMultiModalDataset
from library.model import DualStreamEfficientNet


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)

    # Override Config parameters for a fast baseline execution
    Config.EPOCHS = 4
    Config.BATCH_SIZE = 32
    Config.PATIENCE = 2

    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    print("Using full training data for optimized run...")

    # -------------------------------------------------------------------------
    # 3. Training
    # -------------------------------------------------------------------------
    print("Initializing Engine and starting training...")
    engine = Engine(config=Config)
    best_model_path = engine.run_training()

    # -------------------------------------------------------------------------
    # 4. Full Validation Inference
    # -------------------------------------------------------------------------
    print("\nRunning inference on the full validation set...")

    # Load full validation metadata
    val_df = pd.read_csv(Config.VAL_CSV)

    # Initialize dataset and loader
    # mode="val" ensures caching is used if available (generated during training)
    val_dataset = EEGMultiModalDataset(val_df, Config, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # Load the best model
    device = torch.device(Config.DEVICE)
    model = DualStreamEfficientNet(Config)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    # Inference Loop
    with torch.no_grad():
        for data in val_loader:
            eeg_spec = data["eeg_spec"].to(device)
            kaggle_spec = data["kaggle_spec"].to(device)
            targets = data["target"].to(device)

            with autocast(enabled=Config.USE_AMP):
                logits = model(eeg_spec, kaggle_spec)
                probs = F.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # -------------------------------------------------------------------------
    # 5. Metric Calculation
    # -------------------------------------------------------------------------
    final_metric = kl_divergence_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")

    # Calculate per-sample KL divergence
    epsilon = 1e-15
    preds_clipped = np.clip(all_preds, epsilon, 1 - epsilon)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_target = np.log(all_targets)
        log_pred = np.log(preds_clipped)

        # KL = sum(P * log(P/Q)) = sum(P * log(P)) - sum(P * log(Q))
        # Handle P=0 case where P*log(P)=0
        term1 = np.where(all_targets > 0, all_targets * log_target, 0.0)
        term2 = all_targets * log_pred

        row_kl = np.sum(term1 - term2, axis=1)

    val_df["error_kl"] = row_kl

    # Analyze correlations with metadata features
    features_to_analyze = [
        "total_votes",
        "eeg_label_offset_seconds",
        "spectogram_label_offset_seconds",
    ]

    # Filter for features that actually exist in the dataframe
    features_present = [f for f in features_to_analyze if f in val_df.columns]

    if features_present:
        correlations = (
            val_df[features_present + ["error_kl"]].corr()["error_kl"].drop("error_kl")
        )
        print("Correlation between Error (KL) and Metadata Features:")
        print(correlations)
    else:
        print("No suitable metadata features found for correlation analysis.")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    threshold = 0.8053703308105469

    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is lower than threshold ({threshold}). Generating submission..."
        )
        engine.generate_submission(best_model_path)
    else:
        print(
            f"\nMetric ({final_metric}) is NOT lower than threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
