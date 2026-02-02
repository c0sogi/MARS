import os
import sys
import pandas as pd
import numpy as np
import torch
import torchaudio
import time

# Import from library
from library.config import Config
from library.utils import set_seed, compute_auc
from library.trainer import Trainer
from library.dataset import get_dataloaders


def failure_analysis(val_ids, val_targets, val_preds, df_val):
    """
    Performs failure analysis by correlating errors with audio features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate errors
    errors = np.abs(val_targets - val_preds)

    # Map clip ID to filepath
    id_to_path = pd.Series(
        df_val["filepath"].values, index=df_val["clip"].values
    ).to_dict()

    features_list = []

    print(f"Extracting audio features for {len(val_ids)} validation samples...")

    # Iterate through validation samples to extract features
    for i, clip_id in enumerate(val_ids):
        filepath = id_to_path.get(clip_id)
        if not filepath:
            continue

        full_path = os.path.join(Config.INPUT_ROOT, filepath)

        try:
            # Load audio to extract basic signal stats
            # Note: This adds some overhead but is necessary for the requested analysis
            wav, sr = torchaudio.load(full_path)

            # Features
            duration = wav.shape[1] / sr
            rms = torch.sqrt(torch.mean(wav**2)).item()
            max_amp = torch.max(torch.abs(wav)).item()

            features_list.append(
                {
                    "error": errors[i],
                    "duration": duration,
                    "rms": rms,
                    "max_amp": max_amp,
                    "label": val_targets[i],
                }
            )
        except Exception as e:
            continue

    if not features_list:
        print("No features extracted for analysis.")
        return

    df_analysis = pd.DataFrame(features_list)

    # Compute correlations with error
    correlations = df_analysis.corr()["error"].drop("error")

    print("Correlation between Model Error and Input Features:")
    print(correlations.to_string())

    # Additional insight: Error by class
    print("\nMean Error by Class:")
    print(df_analysis.groupby("label")["error"].mean().to_string())


def main():
    # 1. Configure for Fast Baseline
    # Override Config defaults for this run to ensure it completes within time limits
    Config.EPOCHS = 10

    print(f"Initializing run with {Config.EPOCHS} epochs...")

    # 2. Setup
    set_seed(Config.SEED)

    # Initialize Trainer
    trainer = Trainer()

    # Get DataLoaders (using cached data if available)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Training Loop
    print("Starting Training...")
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = trainer.train_one_epoch(train_loader, epoch)

        # Validate
        val_loss, val_auc = trainer.validate(val_loader)

        # Scheduler Step
        trainer.scheduler.step(val_auc)

        duration = time.time() - start_time
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {duration:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.6f}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(trainer.model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  -> New Best Model Saved (AUC: {best_auc:.6f})")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best AUC: {best_auc:.6f}")

    # 4. Final Evaluation on Best Model
    print("\nLoading best model for final evaluation...")
    if os.path.exists(Config.BEST_MODEL_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
        )
    else:
        print("Warning: Best model not found. Using current model state.")

    trainer.model.eval()

    val_targets = []
    val_preds = []
    val_ids = []

    # Re-run validation inference to get all predictions and IDs aligned
    with torch.no_grad():
        for data, target, ids in val_loader:
            data = data.to(Config.DEVICE)
            output = trainer.model(data)
            probs = torch.sigmoid(output.squeeze(1)).cpu().numpy()

            val_targets.extend(target.cpu().numpy())
            val_preds.extend(probs)
            val_ids.extend(ids)

    val_targets = np.array(val_targets)
    val_preds = np.array(val_preds)

    final_metric = compute_auc(val_targets, val_preds)

    # REQUIRED PRINT
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    df_val = pd.read_csv(Config.VAL_CSV)
    failure_analysis(val_ids, val_targets, val_preds, df_val)

    # 6. Conditional Submission
    THRESHOLD = 0.9942618903292241

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict(test_loader)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
