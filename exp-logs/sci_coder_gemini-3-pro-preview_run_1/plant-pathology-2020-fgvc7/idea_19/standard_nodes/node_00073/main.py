import os
import sys
import importlib
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library modules with reloading to handle persistent environments
import library.config

importlib.reload(library.config)
from library.config import Config

import library.utils

importlib.reload(library.utils)
from library.utils import seed_everything

import library.dataset

importlib.reload(library.dataset)
from library.dataset import AppleDataset, get_transforms

import library.model

importlib.reload(library.model)
from library.model import get_model

import library.loss

importlib.reload(library.loss)

import library.trainer

importlib.reload(library.trainer)
from library import trainer

import library.inference

importlib.reload(library.inference)
from library import inference


def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device
    print(f"Using device: {device}")

    # 2. Data Loading & Preparation
    print("Loading and preparing data...")
    # Load fixed split metadata
    df_full = trainer.get_fixed_split()

    # Calculate Class Weights
    class_weights = trainer.calculate_class_weights(df_full, device)

    # Prepare Validation Data for Ensemble Evaluation
    df_val = df_full[df_full["split"] == "valid"].reset_index(drop=True)
    val_targets = df_val[Config.target_cols].values

    # Initialize array to store Ensemble predictions
    # Shape: (N_val_samples, N_classes)
    val_preds_ensemble = np.zeros((len(df_val), Config.num_classes))

    # Store run scores
    run_scores = []

    # 3. Training Loop (Seed Averaging)
    for run_idx in range(Config.n_runs):
        # Train using the provided trainer module
        # This handles training on fixed train set, validation on fixed val set
        best_auc = trainer.run_training(run_idx, df_full, class_weights)
        run_scores.append(best_auc)

        # ---------------------------------------------------------
        # Generate Predictions on Validation Set for Ensemble
        # ---------------------------------------------------------
        print(f"Generating Validation predictions for Run {run_idx}...")

        # Load the best model checkpoint for this run
        model_path = os.path.join(
            Config.checkpoint_dir, f"{Config.model_name}_run_{run_idx}.pth"
        )
        model = get_model(
            Config.model_name, pretrained=False, num_classes=Config.num_classes
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Create validation dataset and loader
        val_ds = AppleDataset(df_val, transform=get_transforms("valid"))
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Inference loop
        run_probs = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                # Apply Softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)
                run_probs.append(probs.cpu().numpy())

        # Concatenate and add to ensemble accumulator
        run_probs = np.concatenate(run_probs, axis=0)
        val_preds_ensemble += run_probs

    # Average predictions
    val_preds_ensemble /= Config.n_runs

    # 4. Global Evaluation
    print("\nEvaluating Global Performance (Seed Averaging Ensemble)...")

    # Calculate Mean Column-wise ROC AUC
    final_metric = roc_auc_score(val_targets, val_preds_ensemble, average="macro")
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error: 1 - sum(target * prediction)
    # This represents the deviation from the true class probability
    correctness = np.sum(val_targets * val_preds_ensemble, axis=1)
    errors = 1.0 - correctness

    # Extract meta-features for correlation analysis
    widths, heights, intensities = [], [], []

    # Iterate through validation images to extract stats
    for idx, row in df_val.iterrows():
        full_path = os.path.join(Config.input_dir, row["file_path"])
        img = cv2.imread(full_path)

        if img is not None:
            h, w, c = img.shape
            # Calculate mean intensity (normalized 0-1)
            mean_int = img.mean() / 255.0

            widths.append(w)
            heights.append(h)
            intensities.append(mean_int)
        else:
            # Fallback for missing images (should not happen based on checks)
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(
        {"error": errors, "width": widths, "height": heights, "intensity": intensities}
    )

    # Calculate Pearson correlation
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 6. Submission
    threshold = 0.9901680711448418
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({threshold}). Generating submission..."
        )
        inference.generate_submission(
            test_metadata_path=Config.test_metadata_path,
            submission_path=Config.submission_path,
            checkpoint_dir=Config.checkpoint_dir,
            model_name=Config.model_name,
            n_runs=Config.n_runs,
            num_classes=Config.num_classes,
            target_cols=Config.target_cols,
            img_size=Config.img_size,
            batch_size=Config.batch_size,
            num_workers=Config.num_workers,
            device=device,
            seed=Config.seed,
        )
    else:
        print(
            f"\nMetric ({final_metric}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
