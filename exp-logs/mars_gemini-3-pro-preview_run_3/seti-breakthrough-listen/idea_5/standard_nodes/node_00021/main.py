import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.utils import Config, set_seed
from library.model import SiameseSpatialFusionNet
from library.data import TechnosignatureDataset
from library.engine import ModelEngine


def main():
    # --- 1. Configuration & Setup ---
    # Optimized Configuration
    Config.NUM_EPOCHS = 12
    Config.BATCH_SIZE = 64

    # Initialize Engine (sets up Model, Optimizer, Scheduler, Seed, Directories)
    engine = ModelEngine()

    print(
        f"Configuration: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={engine.device}"
    )

    # --- 2. Data Loading ---
    # Create Datasets
    train_ds = TechnosignatureDataset(
        os.path.join(Config.METADATA_DIR, "train.csv"), data_type="train"
    )
    val_ds = TechnosignatureDataset(
        os.path.join(Config.METADATA_DIR, "val.csv"), data_type="val"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- 3. Training Loop ---
    best_auc = 0.0
    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train one epoch
        train_loss = engine.train_one_epoch(train_loader)

        # Validate
        val_loss, val_auc = engine.validate(val_loader)

        # Update Scheduler
        engine.scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(engine.model.state_dict(), engine.best_model_path)

    print(f"Training finished. Best Val AUC: {best_auc}")

    # --- 4. Final Validation & Failure Analysis ---
    print("\nRunning Final Validation and Failure Analysis...")

    # Load best model weights
    engine.model.load_state_dict(
        torch.load(engine.best_model_path, map_location=engine.device)
    )
    engine.model.eval()

    # Containers for analysis
    all_targets = []
    all_preds = []

    # Feature accumulators
    feat_mean_on = []
    feat_std_on = []
    feat_mean_off = []
    feat_std_off = []
    feat_mean_diff = []

    with torch.no_grad():
        for images, targets in val_loader:
            img_on, img_off = images
            img_on = img_on.to(engine.device)
            img_off = img_off.to(engine.device)

            # Inference
            outputs = engine.model(img_on, img_off)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy())

            # Extract simple image statistics for failure analysis
            # Compute stats per sample in the batch
            # img_on shape: (B, 3, H, W) -> mean over (1,2,3)
            m_on = img_on.mean(dim=[1, 2, 3]).cpu().numpy()
            s_on = img_on.std(dim=[1, 2, 3]).cpu().numpy()
            m_off = img_off.mean(dim=[1, 2, 3]).cpu().numpy()
            s_off = img_off.std(dim=[1, 2, 3]).cpu().numpy()

            feat_mean_on.extend(m_on)
            feat_std_on.extend(s_on)
            feat_mean_off.extend(m_off)
            feat_std_off.extend(s_off)
            feat_mean_diff.extend(m_on - m_off)

    # Compute Final Metric
    final_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation
    df_analysis = pd.DataFrame(
        {
            "target": all_targets,
            "pred": all_preds,
            "mean_on": feat_mean_on,
            "std_on": feat_std_on,
            "mean_off": feat_mean_off,
            "std_off": feat_std_off,
            "mean_diff": feat_mean_diff,
        }
    )

    # Calculate Error Magnitude
    df_analysis["error"] = (df_analysis["target"] - df_analysis["pred"]).abs()

    print("Correlation between Error Magnitude and Input Features:")
    correlations = df_analysis.corr()["error"].drop(["error", "target", "pred"])
    print(correlations)

    # --- 5. Submission Generation ---
    THRESHOLD = 0.7770832449065452

    if final_auc > THRESHOLD:
        print(
            f"\nMetric ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_metadata_path = os.path.join(Config.METADATA_DIR, "test.csv")
        test_ds = TechnosignatureDataset(test_metadata_path, data_type="test")
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Predict using TTA (implemented in engine)
        preds = engine.predict_with_tta(test_loader)

        # Create Submission DataFrame
        df_test = pd.read_csv(test_metadata_path)
        df_test["target"] = preds
        submission_df = df_test[["id", "target"]]

        # Save
        os.makedirs("./submission", exist_ok=True)
        save_path = "./submission/submission.csv"
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(f"\nMetric ({final_auc}) <= Threshold ({THRESHOLD}). Submission skipped.")


if __name__ == "__main__":
    main()
