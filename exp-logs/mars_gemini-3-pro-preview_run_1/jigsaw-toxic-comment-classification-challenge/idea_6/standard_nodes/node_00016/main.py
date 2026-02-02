import os
import sys
import numpy as np
import torch
import warnings
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.dataset import ToxicityDataset
from library.model import ToxicityModel
from library.trainer import Trainer
from library.utils import seed_everything, get_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Configuration
    # Ensure reproducibility
    seed_everything(Config.seed)

    # Override Config for a fast baseline execution
    # We limit epochs to 1 to ensure runtime is well within the 2-hour limit
    # while still utilizing the full dataset for a strong baseline.
    Config.epochs = 1

    print(
        f"Starting execution with {Config.epochs} epoch(s) on device: {Config.device}"
    )

    # 2. Training
    trainer = Trainer()
    # Fit the model (handles training, validation, and saving best model)
    trainer.fit(load_cached_data=True)

    # 3. Validation & Failure Analysis
    print("Starting Validation and Failure Analysis...")

    # Load Validation Data
    val_dataset = ToxicityDataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
    )

    # Load Best Model
    model = ToxicityModel()
    if not os.path.exists(trainer.model_path):
        print("Model file not found. Training may have failed.")
        return

    model.load_state_dict(torch.load(trainer.model_path, map_location=Config.device))
    model.to(Config.device)
    model.eval()

    all_preds = []
    all_targets = []
    all_lengths = []

    # Inference Loop
    with torch.no_grad():
        for data in val_loader:
            ids = data["input_ids"].to(Config.device, dtype=torch.long)
            mask = data["attention_mask"].to(Config.device, dtype=torch.long)
            targets = data["labels"].to(Config.device, dtype=torch.float)

            # Forward pass
            outputs = model(ids, mask)
            preds = torch.sigmoid(outputs)

            # Collect data
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Calculate lengths from attention mask (sum of 1s)
            # Shape: (batch_size,)
            lengths = mask.sum(dim=1).cpu().numpy()
            all_lengths.append(lengths)

    # Concatenate results
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    seq_lengths = np.concatenate(all_lengths)

    # Calculate Metric
    final_score = get_score(y_true, y_pred)
    # Print full precision as requested
    print(f"Final Validation Metric: {final_score}")

    # Failure Analysis
    # Calculate Mean Absolute Error per sample (average error across the 6 classes)
    # Shape: (N,)
    error_magnitude = np.abs(y_true - y_pred).mean(axis=1)

    # Calculate Pearson Correlation
    correlation, _ = pearsonr(error_magnitude, seq_lengths)
    print(f"Correlation between Error Magnitude and Token Length: {correlation}")

    # 4. Submission Generation
    # Strict threshold check
    THRESHOLD = 0.9920650979347099

    if final_score > THRESHOLD:
        print(
            f"Validation metric {final_score} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        trainer.predict(load_cached_data=True)
    else:
        print(
            f"Validation metric {final_score} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
