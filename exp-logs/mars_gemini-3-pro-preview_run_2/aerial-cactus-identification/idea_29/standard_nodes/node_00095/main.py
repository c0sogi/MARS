import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import library functions
from library.utils import seed_everything, calculate_roc_auc
from library.dataset import load_data, CactusDataset, get_transforms
from library.model import WideAntiAliasedRes2NeXt
from library.train import run_training

# Configuration
WORK_DIR = "./working/idea_29"
SUBMISSION_PATH = "./submission/submission.csv"
EPOCHS = 10
BATCH_SIZE = 64
SEEDS = [0, 1, 2, 3, 4]


def main():
    # Set seed for reproducibility
    seed_everything(42)

    print("Starting training pipeline...")

    # 1. Train the model ensemble and generate submission
    # run_training handles training 5 seeds and generating the submission file with TTA
    run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        weight_decay=1e-4,
        n_seeds=len(SEEDS),
        debug=False,
        work_dir=WORK_DIR,
        submission_path=SUBMISSION_PATH,
    )

    # 2. Validation & Failure Analysis
    print("\nStarting Validation and Failure Analysis...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation data
    # load_data returns (images, labels, ids)
    val_imgs, val_lbls, val_ids = load_data("val", load_cached_data=True)

    # Create DataLoader for validation
    val_dataset = CactusDataset(val_imgs, val_lbls, transform=get_transforms("val"))
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Generate Ensemble Predictions on Validation Set
    # We need to load each model and average predictions
    ensemble_preds = np.zeros(len(val_lbls))
    models_found = 0

    for seed in SEEDS:
        model_path = os.path.join(WORK_DIR, f"model_seed_{seed}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model for seed {seed} not found.")
            continue

        # Load model
        model = WideAntiAliasedRes2NeXt()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Predict using TTA to match test set methodology
        seed_preds = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                # TTA: Original + Flip H + Flip V
                out = torch.sigmoid(model(images))
                out_h = torch.sigmoid(model(torch.flip(images, [3])))
                out_v = torch.sigmoid(model(torch.flip(images, [2])))
                p = (out + out_h + out_v) / 3.0
                seed_preds.extend(p.cpu().numpy().flatten())

        ensemble_preds += np.array(seed_preds)
        models_found += 1

    if models_found > 0:
        ensemble_preds /= models_found

    # Calculate Final Metric
    final_auc = calculate_roc_auc(val_lbls, ensemble_preds)
    # Print exactly as requested
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    # Calculate error magnitude
    errors = np.abs(val_lbls - ensemble_preds)

    print(
        "\nFailure Analysis (Correlation between Error Magnitude and Image Meta-Features):"
    )

    # Compute meta-features for validation images
    # val_imgs is (N, 32, 32, 3)
    # Vectorized computation for speed

    brightness = np.mean(val_imgs, axis=(1, 2, 3))
    contrast = np.std(val_imgs, axis=(1, 2, 3))
    red_mean = np.mean(val_imgs[:, :, :, 0], axis=(1, 2))
    green_mean = np.mean(val_imgs[:, :, :, 1], axis=(1, 2))
    blue_mean = np.mean(val_imgs[:, :, :, 2], axis=(1, 2))

    features = {
        "Brightness": brightness,
        "Contrast": contrast,
        "Red Mean": red_mean,
        "Green Mean": green_mean,
        "Blue Mean": blue_mean,
    }

    for name, values in features.items():
        # Pearson correlation
        corr, pval = pearsonr(values, errors)
        print(f"{name}: Correlation = {corr:.4f} (p-value = {pval:.4f})")


if __name__ == "__main__":
    main()
