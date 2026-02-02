import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
import importlib
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library modules
import library.config
import library.utils
import library.dataset
import library.model
import library.trainer
import library.inference

# Reload modules to handle persistent session caching
importlib.reload(library.config)
importlib.reload(library.utils)
importlib.reload(library.dataset)
importlib.reload(library.model)
importlib.reload(library.trainer)
importlib.reload(library.inference)

from library.config import Config
from library.utils import seed_everything
from library.dataset import AppleDataset, get_transforms
from library.model import get_model
from library import trainer
from library import inference


def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device
    print(f"Using device: {device}")

    # 2. Data Loading & Preparation
    print("Loading data for Fixed-Split Seed Averaging...")
    # Load separate train and val sets (80/20 split)
    df_train, df_val = trainer.get_data_split()

    # Calculate Class Weights on training set
    class_weights = trainer.calculate_class_weights(df_train, device)

    # Initialize array to store Ensemble predictions on Validation Set
    # Shape: (N_val_samples, N_classes)
    ensemble_val_preds = np.zeros((len(df_val), Config.num_classes))

    # 3. Training Loop (Seeds)
    for seed in Config.seeds:
        # Train the model for this seed and get best validation predictions
        val_preds = trainer.run_seed(seed, df_train, df_val, class_weights)

        # Accumulate predictions for averaging
        ensemble_val_preds += val_preds

    # Average the predictions
    ensemble_val_preds /= len(Config.seeds)

    # 4. Global Evaluation (Ensemble Metric)
    print("\nEvaluating Ensemble Performance on Hold-Out Set...")
    # Extract ground truth targets from validation set
    val_targets = df_val[Config.target_cols].values

    # Calculate Mean Column-wise ROC AUC
    final_metric = roc_auc_score(val_targets, ensemble_val_preds, average="macro")
    print(f"Final Validation Metric (Ensemble): {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error: 1 - sum(target * prediction)
    correctness = np.sum(val_targets * ensemble_val_preds, axis=1)
    errors = 1.0 - correctness

    # Extract meta-features for correlation analysis on Validation Set
    widths, heights, intensities = [], [], []

    for idx, row in df_val.iterrows():
        full_path = os.path.join(Config.input_dir, row["file_path"])
        img = cv2.imread(full_path)

        if img is not None:
            h, w, c = img.shape
            mean_int = img.mean() / 255.0

            widths.append(w)
            heights.append(h)
            intensities.append(mean_int)
        else:
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
        # Note: generate_submission in inference.py has been updated to use Config.seeds
        inference.generate_submission(
            test_metadata_path=Config.test_metadata_path,
            submission_path=Config.submission_path,
            checkpoint_dir=Config.checkpoint_dir,
            model_name=Config.model_name,
            n_folds=0,  # Unused in updated function
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
