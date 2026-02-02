import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library modules
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
    print("Loading and preparing data...")
    # Merge train and validation metadata for full cross-validation
    df_full = trainer.get_all_data()

    # Generate Stratified Folds
    df_folds = trainer.get_folds(df_full, n_folds=Config.n_folds, seed=Config.seed)

    # Calculate Class Weights
    class_weights = trainer.calculate_class_weights(df_folds, device)

    # Initialize array to store Out-Of-Fold (OOF) predictions
    # Shape: (N_samples, N_classes)
    oof_preds = np.zeros((len(df_full), Config.num_classes))

    # Store fold scores
    fold_scores = []

    # 3. Training Loop (5 Folds)
    for fold_idx in range(Config.n_folds):
        # Train the fold using the provided trainer module
        # This handles training, validation, and saving the best checkpoint
        best_auc = trainer.run_fold(fold_idx, df_folds, class_weights)
        fold_scores.append(best_auc)

        # ---------------------------------------------------------
        # Generate OOF Predictions for this fold
        # ---------------------------------------------------------
        print(f"Generating OOF predictions for Fold {fold_idx}...")

        # Load the best model checkpoint for this fold
        model_path = os.path.join(
            Config.checkpoint_dir, f"{Config.model_name}_fold_{fold_idx}.pth"
        )
        model = get_model(
            Config.model_name, pretrained=False, num_classes=Config.num_classes
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Get validation data for this fold
        val_df = df_folds[df_folds["fold"] == fold_idx].reset_index(drop=True)
        val_indices = df_folds[df_folds["fold"] == fold_idx].index

        # Create validation dataset and loader
        val_ds = AppleDataset(val_df, transform=get_transforms("valid"))
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Inference loop
        fold_probs = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                # Apply Softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)
                fold_probs.append(probs.cpu().numpy())

        # Concatenate and store predictions
        fold_probs = np.concatenate(fold_probs, axis=0)
        oof_preds[val_indices] = fold_probs

    # 4. Global Evaluation
    print("\nEvaluating Global Performance...")
    # Extract ground truth targets
    oof_targets = df_folds[Config.target_cols].values

    # Calculate Mean Column-wise ROC AUC
    final_metric = roc_auc_score(oof_targets, oof_preds, average="macro")
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error: 1 - sum(target * prediction)
    # This represents the deviation from the true class probability
    correctness = np.sum(oof_targets * oof_preds, axis=1)
    errors = 1.0 - correctness

    # Extract meta-features for correlation analysis
    widths, heights, intensities = [], [], []

    # Iterate through all images to extract stats
    # Using cv2 to quickly read dimensions and mean intensity
    for idx, row in df_folds.iterrows():
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
            n_folds=Config.n_folds,
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
