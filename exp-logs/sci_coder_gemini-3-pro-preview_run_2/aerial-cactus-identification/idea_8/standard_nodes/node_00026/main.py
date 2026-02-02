import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import from library
from library.utils import set_seed, get_device
from library.dataset import get_dataloaders
from library.train import run_training
from library.predict import generate_submission
from library.model import LightweightPyramidNet

# --- Configuration ---
SEEDS = [0, 1, 2, 3, 4]
WORKING_DIR = "./working/optimized"
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
EPOCHS = 12  # Reduced epochs as model converges fast. Cite solution_lesson_node_00019
BATCH_SIZE = 256  # Increased for efficiency on A100
PATIENCE = 5
LR = 2e-3  # Scaled for larger batch size
WEIGHT_DECAY = 1e-2


def main():
    # 1. Setup
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Training Loop (Homogeneous Ensemble)
    model_paths = []
    print("========================================")
    print("           STARTING TRAINING            ")
    print("========================================")

    for seed in SEEDS:
        print(f"\n--- Training Seed {seed} ---")
        path, best_auc = run_training(
            seed=seed,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            save_dir=WORKING_DIR,
        )
        model_paths.append(path)

    # 3. Validation & Ensemble Evaluation
    print("\n========================================")
    print("           VALIDATION & METRICS         ")
    print("========================================")

    val_loader, val_labels, val_preds = get_ensemble_predictions(model_paths, device)

    # Compute Metric (AUC)
    final_auc = roc_auc_score(val_labels, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 4. Failure Analysis
    print("\n========================================")
    print("           FAILURE ANALYSIS             ")
    print("========================================")
    perform_failure_analysis(val_loader, val_labels, val_preds)

    # 5. Submission
    # Note: The requirement "higher than 1.0" for AUC is mathematically impossible (max AUC is 1.0).
    # We assume this is a template error and proceed if the model is better than random guessing (> 0.5).
    if final_auc > 0.5:
        print("\n========================================")
        print("           GENERATING SUBMISSION        ")
        print("========================================")
        generate_submission(
            model_paths=model_paths,
            output_file=SUBMISSION_FILE,
            batch_size=BATCH_SIZE,
            num_workers=4,
            seed=42,
        )
    else:
        print("\nValidation metric too low. Skipping submission.")


def get_ensemble_predictions(model_paths, device):
    """
    Computes ensemble predictions on the validation set.
    """
    # Get dataloaders
    # We use load_cached_data=True as requested
    _, val_loader, _, _ = get_dataloaders(
        batch_size=BATCH_SIZE, load_cached_data=True, seed=42
    )

    # 1. Collect Ground Truth Labels
    # Although dataloader is deterministic, we iterate to be sure of alignment
    all_targets = []
    for _, labels in val_loader:
        all_targets.append(labels.numpy())
    all_targets = np.concatenate(all_targets)

    num_samples = len(all_targets)
    accumulated_probs = np.zeros(num_samples, dtype=np.float64)

    # 2. Accumulate Predictions from each Model
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Warning: Model {path} not found.")
            continue

        model = LightweightPyramidNet().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        preds = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)

                # Standard inference (no TTA for validation speed)
                logits = model(images)
                probs = torch.sigmoid(logits)
                preds.append(probs.cpu().numpy())

        accumulated_probs += np.concatenate(preds).flatten()

        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average
    avg_probs = accumulated_probs / len(model_paths)

    return val_loader, all_targets, avg_probs


def perform_failure_analysis(val_loader, labels, preds):
    """
    Analyzes correlations between prediction error and image features.
    """
    # Calculate Absolute Error
    errors = np.abs(labels - preds)

    # Extract Image Meta-Features
    # We iterate the loader again. It's fast for this dataset size.
    brightness = []
    contrast = []
    red_mean = []
    green_mean = []
    blue_mean = []

    for images, _ in val_loader:
        # images is Tensor (B, C, H, W) in [0, 1]
        # Convert to numpy (B, H, W, C) for stats
        imgs_np = images.permute(0, 2, 3, 1).numpy()

        for img in imgs_np:
            # img is (H, W, C)
            brightness.append(np.mean(img))
            contrast.append(np.std(img))
            red_mean.append(np.mean(img[:, :, 0]))
            green_mean.append(np.mean(img[:, :, 1]))
            blue_mean.append(np.mean(img[:, :, 2]))

    # Create Analysis DataFrame
    df = pd.DataFrame(
        {
            "error": errors,
            "brightness": brightness,
            "contrast": contrast,
            "red_mean": red_mean,
            "green_mean": green_mean,
            "blue_mean": blue_mean,
        }
    )

    print("Correlation between Error Magnitude and Image Features:")
    features = ["brightness", "contrast", "red_mean", "green_mean", "blue_mean"]

    for feat in features:
        # Calculate Pearson Correlation
        corr, _ = pearsonr(df["error"], df[feat])
        print(f"{feat.ljust(12)}: {corr:.4f}")


if __name__ == "__main__":
    main()
