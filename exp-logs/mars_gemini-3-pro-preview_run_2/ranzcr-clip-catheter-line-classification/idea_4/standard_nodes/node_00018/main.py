import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from the provided library
from library.config import Config
from library.train import run_training
from library.predict import inference
from library.dataset import CatheterDataset, get_transforms
from library.model import CatheterModel
from library.utils import seed_everything, get_score


def main():
    # --- 1. Configuration & Setup ---
    seed_everything(Config.seed)

    # Use full dataset and configured epochs (Cite solution_lesson_node_00017)
    print(f"Training with full dataset: {Config.train_metadata_path}")

    # --- 2. Training ---
    print("\n=== Starting Training ===")
    # run_training handles the loop and saves 'best_model.pth' in Config.working_dir
    # It returns the best validation score achieved during training
    _ = run_training()

    # --- 3. Validation & Failure Analysis ---
    print("\n=== Starting Validation & Failure Analysis ===")

    device = torch.device(Config.device)

    # Load Validation Data
    df_val = pd.read_csv(Config.val_metadata_path)
    val_dataset = CatheterDataset(df_val, transform=get_transforms("valid"))
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Load Best Model
    model = CatheterModel(
        model_name=Config.model_name, pretrained=False, num_classes=Config.num_classes
    )
    best_model_path = os.path.join(Config.working_dir, "best_model.pth")

    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded best model for analysis.")
    else:
        print("Error: Best model not found.")
        return

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    # Inference Loop
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, dtype=torch.float)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Final Metric
    final_score = get_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_score:.16f}")

    # Failure Analysis
    # Calculate Mean Absolute Error per sample
    # Shape: (N_samples, N_classes) -> mean over classes -> (N_samples,)
    errors = np.abs(all_targets - all_preds).mean(axis=1)

    # Feature: Number of active labels (Label Density)
    label_density = all_targets.sum(axis=1)

    # Correlation
    if len(errors) > 1:
        corr, p_value = pearsonr(label_density, errors)
        print(f"Correlation between Error Magnitude and Label Density: {corr:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # --- 4. Submission ---
    threshold = 0.9539590105428797

    if final_score > threshold:
        print(
            f"\nValidation score ({final_score:.6f}) exceeds threshold ({threshold:.6f}). Generating submission..."
        )
        submission_path = "./submission/submission.csv"

        # Use the provided inference function from library.predict
        # We need to ensure it uses the correct model path
        inference(
            model_path=best_model_path,
            output_path=submission_path,
            batch_size=Config.batch_size,
            device=Config.device,
            num_workers=Config.num_workers,
        )
    else:
        print(
            f"\nValidation score ({final_score:.6f}) did not meet threshold ({threshold:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
