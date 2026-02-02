import os
import sys
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.dataset import AudioDataset, collate_fn
from library.trainer import Trainer
from library.utils import set_seed, load_checkpoint, calculate_lrap


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Fast Baseline Configuration:
    # Limit epochs to ensure execution finishes quickly (< 2 hours).
    # The A100 GPU can handle the full dataset (23k samples) efficiently,
    # so we reduce epochs rather than data size to maintain performance.
    # Increased to 25 epochs to allow continuous Mixup to converge (Cite solution_lesson_node_00014)
    Config.EPOCHS = 25

    print(f"Running on device: {Config.DEVICE}")
    print(f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n--- Initializing Data Loaders ---")
    train_dataset = AudioDataset(mode="train")
    val_dataset = AudioDataset(mode="val")
    test_dataset = AudioDataset(mode="test")

    print(f"Training Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")
    print(f"Test Samples: {len(test_dataset)}")

    # Create DataLoaders
    # pin_memory=True speeds up host-to-device transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ==========================================
    # 3. Training
    # ==========================================
    print("\n--- Starting Training ---")
    trainer = Trainer(train_loader, val_loader, test_loader)

    # Train with early stopping
    trainer.train(patience=3)

    # ==========================================
    # 4. Final Validation & Failure Analysis
    # ==========================================
    print("\n--- Final Validation & Failure Analysis ---")

    # Load the best model checkpoint to ensure we evaluate the optimal state
    if os.path.exists(trainer.best_model_path):
        print(f"Loading best model from: {trainer.best_model_path}")
        load_checkpoint(trainer.best_model_path, trainer.model, device=trainer.device)
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    # Ensure model is in evaluation mode
    trainer.model.eval()

    all_targets = []
    all_preds = []

    # Run Inference on Validation Set
    # We use no_grad to save memory and computation
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(trainer.device)
            # Targets are needed for metric calculation

            # Forward pass
            outputs = trainer.model(data)
            probs = torch.sigmoid(outputs)

            all_targets.append(target.numpy())
            all_preds.append(probs.cpu().numpy())

    # Concatenate results
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Calculate Final Metric (LWLRAP)
    final_metric = calculate_lrap(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("Performing failure analysis...")

    # Calculate Mean Absolute Error (MAE) per sample across all classes
    # High MAE indicates the model's probabilities diverged significantly from ground truth
    mae_per_sample = np.mean(np.abs(all_targets - all_preds), axis=1)

    # Extract Metadata Features (Duration and Label Count)
    # We read audio headers to get duration for the validation set
    durations = []
    for idx, row in val_dataset.df.iterrows():
        filepath = os.path.join(Config.INPUT_ROOT, row["filepath"])
        try:
            info = sf.info(filepath)
            durations.append(info.duration)
        except Exception:
            durations.append(0.0)

    durations = np.array(durations)
    label_counts = np.sum(all_targets, axis=1)

    # Calculate Correlations
    if len(durations) == len(mae_per_sample):
        corr_duration = np.corrcoef(mae_per_sample, durations)[0, 1]
        corr_label_count = np.corrcoef(mae_per_sample, label_counts)[0, 1]

        print(f"Correlation (Error vs Duration): {corr_duration:.6f}")
        print(f"Correlation (Error vs Label Count): {corr_label_count:.6f}")
    else:
        print("Error: Metadata length mismatch. Skipping correlation analysis.")

    # ==========================================
    # 5. Submission
    # ==========================================
    TARGET_THRESHOLD = 0.8102296452969313

    if final_metric > TARGET_THRESHOLD:
        print(f"\nMetric {final_metric} > {TARGET_THRESHOLD}. Generating submission...")
        trainer.predict()
    else:
        print(f"\nMetric {final_metric} <= {TARGET_THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
