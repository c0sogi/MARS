import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.trainer import Trainer
from library.dataset import EEGDataset
from library.model import MultiResDualStreamNet
from library.utils import seed_everything, kl_divergence_score
from library.inference import predict


def run():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Configure for a fast baseline execution as requested
    Config.EPOCHS = 2
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 5000  # Limit training to 5000 samples for speed

    # Ensure reproducibility
    seed_everything(Config.SEED)

    print(
        f"Starting execution with EPOCHS={Config.EPOCHS}, SUBSET_SIZE={Config.DEBUG_SUBSET_SIZE}"
    )

    # ==========================================
    # 2. Model Training
    # ==========================================
    print("\n--- Starting Training ---")
    # Initialize trainer with debug=True to use the subset configuration
    trainer = Trainer(config=Config, debug=True)
    trainer.fit()

    # ==========================================
    # 3. Full Validation
    # ==========================================
    print("\n--- Starting Full Validation ---")
    device = torch.device(Config.DEVICE)

    # Load the best model saved during training
    # We use pretrained=False because we are loading custom weights immediately
    model = MultiResDualStreamNet(pretrained=False)
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Initialize Full Validation Dataset (No subsetting)
    val_dataset = EEGDataset(
        mode="val", config=Config, load_cached_data=True, subset_size=None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # Inference Loop
    all_preds = []
    all_targets = []

    print(f"Validating on {len(val_dataset)} samples...")
    with torch.no_grad():
        for (x_a, x_b), targets in val_loader:
            x_a = x_a.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)

            # Forward pass
            logits = model((x_a, x_b))
            probs = F.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Final Metric
    final_metric = kl_divergence_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n--- Performing Failure Analysis ---")

    # Calculate KL Divergence per sample
    # KL(P || Q) = sum(P * log(P/Q))
    epsilon = 1e-15
    y_pred = np.clip(all_preds, epsilon, 1 - epsilon)
    y_true = all_targets

    # Term 1: P * log(P)
    term1 = np.zeros_like(y_true)
    mask = y_true > 0
    term1[mask] = y_true[mask] * np.log(y_true[mask])

    # Term 2: P * log(Q)
    term2 = y_true * np.log(y_pred)

    # Sum over classes
    kl_per_sample = np.sum(term1 - term2, axis=1)

    # Load Validation Metadata
    val_df = pd.read_csv(Config.VAL_CSV)

    if len(val_df) != len(kl_per_sample):
        print("Error: Validation metadata length does not match predictions length.")
    else:
        val_df["error_magnitude"] = kl_per_sample

        # Features to correlate
        features = [
            "total_votes",
            "eeg_label_offset_seconds",
            "spectogram_label_offset_seconds",
        ]

        print("Correlation between Error Magnitude and Metadata Features:")
        for feat in features:
            if feat in val_df.columns:
                corr = val_df[feat].corr(val_df["error_magnitude"])
                print(f"  {feat}: {corr}")
            else:
                print(f"  {feat}: Not found in metadata")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    print("\n--- Checking Submission Criteria ---")
    threshold = 0.8169508603799445

    if final_metric < threshold:
        print(
            f"Metric {final_metric} is lower than threshold {threshold}. Generating submission..."
        )
        submission_path = "./submission/submission.csv"
        predict(
            config=Config, checkpoint_path=checkpoint_path, output_path=submission_path
        )
    else:
        print(
            f"Metric {final_metric} is not lower than threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    run()
