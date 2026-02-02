import os
import sys
import cv2
import torch
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss

# Import provided library components
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.engine import train_fold
from library.model import DogClassifier
from library.data import get_dataloaders, get_test_dataloader

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    print("Initializing Run...")
    seed_everything(Config.seed)
    device = torch.device(Config.device)

    # Ensure working directory exists (handled by Config.setup usually, but good to be safe)
    os.makedirs(Config.working_dir, exist_ok=True)
    os.makedirs(Config.submission_dir, exist_ok=True)

    # 2. Training Loop (5 Folds)
    print(f"Starting Training for {Config.n_folds} Folds...")
    # We use the full 30 epochs as defined in Config to ensure convergence for Model Soups
    for fold in range(Config.n_folds):
        train_fold(fold)

    # 3. Validation & Failure Analysis
    print("\nStarting Validation and Failure Analysis...")

    oof_preds = []
    oof_targets = []
    meta_stats = []

    # We need the class list for submission later
    classes = None

    for fold in range(Config.n_folds):
        print(f"Evaluating Fold {fold}...")

        # Load Data
        _, val_loader, fold_classes = get_dataloaders(fold)
        if classes is None:
            classes = fold_classes

        # Load Best Model (Greedy Soup)
        model = DogClassifier(pretrained=False)
        model_path = os.path.join(Config.working_dir, f"best_model_fold_{fold}.pth")

        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            continue

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Inference
        fold_probs = []
        fold_labels = []

        # For failure analysis
        fold_file_sizes = []
        fold_aspect_ratios = []

        # Access dataset to get file paths for feature extraction
        val_dataset = val_loader.dataset

        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)

                # Forward pass
                logits = model(images)
                probs = torch.softmax(logits, dim=1)

                fold_probs.append(probs.cpu().numpy())
                fold_labels.append(labels.numpy())

        # Concatenate fold results
        fold_probs = np.concatenate(fold_probs)
        fold_labels = np.concatenate(fold_labels)

        oof_preds.append(fold_probs)
        oof_targets.append(fold_labels)

        # Extract metadata features for failure analysis
        # We do this by iterating the dataset indices corresponding to the loader
        # Since loader is not shuffled (val), we can iterate dataset directly or use paths
        # However, we need to be careful with drop_last=False (default for val).
        # The safest way is to iterate the dataset using the same order.

        print(f"Extracting metadata features for Fold {fold}...")
        for idx in range(len(val_dataset)):
            fp = val_dataset.file_paths[idx]

            # File Size
            try:
                f_size = os.path.getsize(fp)
            except:
                f_size = 0
            fold_file_sizes.append(f_size)

            # Aspect Ratio (Read image)
            try:
                img = cv2.imread(fp)
                if img is not None:
                    h, w = img.shape[:2]
                    ar = w / h if h > 0 else 0
                else:
                    ar = 0
            except:
                ar = 0
            fold_aspect_ratios.append(ar)

        # Calculate per-sample loss
        # Select probability of true class
        true_probs = fold_probs[np.arange(len(fold_labels)), fold_labels]
        # Clip to avoid log(0)
        true_probs = np.clip(true_probs, 1e-15, 1 - 1e-15)
        sample_losses = -np.log(true_probs)

        # Store stats
        fold_stats = pd.DataFrame(
            {
                "loss": sample_losses,
                "file_size": fold_file_sizes,
                "aspect_ratio": fold_aspect_ratios,
            }
        )
        meta_stats.append(fold_stats)

    # Aggregate OOF
    all_oof_preds = np.concatenate(oof_preds)
    all_oof_targets = np.concatenate(oof_targets)
    full_meta_stats = pd.concat(meta_stats, ignore_index=True)

    # Compute Final Metric
    final_metric = calculate_metric(all_oof_targets, all_oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    corr_size, _ = spearmanr(full_meta_stats["loss"], full_meta_stats["file_size"])
    corr_ar, _ = spearmanr(full_meta_stats["loss"], full_meta_stats["aspect_ratio"])

    print("Failure Analysis - Error Correlations:")
    print(f"  Correlation (Error vs File Size): {corr_size:.4f}")
    print(f"  Correlation (Error vs Aspect Ratio): {corr_ar:.4f}")

    # 4. Submission
    THRESHOLD = 0.14004325100369866

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating Submission...")

        test_loader, test_ids = get_test_dataloader()
        num_test_samples = len(test_ids)
        avg_preds = np.zeros((num_test_samples, Config.num_classes))

        # Ensemble Inference
        for fold in range(Config.n_folds):
            print(f"Inference with Fold {fold} Model...")
            model = DogClassifier(pretrained=False)
            model_path = os.path.join(Config.working_dir, f"best_model_fold_{fold}.pth")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            fold_preds = []

            with torch.no_grad():
                for images in test_loader:
                    images = images.to(device)

                    # TTA: Original
                    logits1 = model(images)
                    probs1 = torch.softmax(logits1, dim=1)

                    # TTA: Horizontal Flip
                    images_flip = torch.flip(images, [3])
                    logits2 = model(images_flip)
                    probs2 = torch.softmax(logits2, dim=1)

                    # Average
                    avg_prob = (probs1 + probs2) / 2
                    fold_preds.append(avg_prob.cpu().numpy())

            avg_preds += np.concatenate(fold_preds)

        # Average across folds
        avg_preds /= Config.n_folds

        # Create Submission DataFrame
        sub_df = pd.DataFrame(avg_preds, columns=classes)
        sub_df.insert(0, "id", test_ids)

        sub_df.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")

    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
