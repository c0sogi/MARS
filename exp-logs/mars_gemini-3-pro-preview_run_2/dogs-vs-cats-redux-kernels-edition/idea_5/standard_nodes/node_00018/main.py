import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from PIL import Image

# Import from provided libraries
from library.config import Config
from library.utils import set_seed
from library.data import get_folded_data, get_train_val_loaders
from library.model import get_model
from library.trainer import train_fold
from library.inference import run_inference


def run_oof_inference(fold_idx, val_loader, device):
    """
    Runs inference on the validation set using the best checkpoint for the given fold.
    Returns probabilities and targets.
    """
    # Initialize model
    model = get_model(pretrained=False, device=device)

    # Load checkpoint
    checkpoint_path = os.path.join(Config.checkpoint_dir, f"fold_{fold_idx}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint for fold {fold_idx} not found.")
        return np.array([]), np.array([])

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for images, target in val_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)

            preds.append(probs.cpu().numpy())
            targets.append(target.numpy())

    return np.concatenate(preds), np.concatenate(targets)


def perform_failure_analysis(results_df):
    """
    Calculates correlations between error magnitude and image metadata.
    """
    print("\n=== Failure Analysis ===")

    # Calculate error magnitude
    results_df["error"] = (results_df["prob"] - results_df["label"]).abs()

    # Collect image metadata
    # We need to read image dimensions. To be fast, we only read headers.
    widths = []
    heights = []
    aspect_ratios = []

    print("Collecting image metadata for failure analysis...")
    for filepath in results_df["filepath"]:
        full_path = os.path.join(Config.input_dir, filepath)
        try:
            with Image.open(full_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
            widths.append(np.nan)
            heights.append(np.nan)
            aspect_ratios.append(np.nan)

    results_df["width"] = widths
    results_df["height"] = heights
    results_df["aspect_ratio"] = aspect_ratios

    # Calculate correlations
    correlations = results_df[["error", "width", "height", "aspect_ratio"]].corr()[
        "error"
    ]

    print("Correlation between Error Magnitude and Input Features:")
    print(f"  Width: {correlations['width']:.4f}")
    print(f"  Height: {correlations['height']:.4f}")
    print(f"  Aspect Ratio: {correlations['aspect_ratio']:.4f}")

    return correlations


def main():
    # 1. Setup
    set_seed(Config.seed)
    device = Config.device
    print(f"Using device: {device}")

    # 2. Prepare Data
    # This generates/loads the folds parquet file
    folds_df = get_folded_data(load_cached_data=True)

    # Containers for global validation
    all_oof_preds = []
    all_oof_targets = []
    all_oof_filepaths = []

    # 3. Training Loop (5 Folds)
    for fold_idx in range(Config.n_folds):
        print(f"\n--- Processing Fold {fold_idx}/{Config.n_folds - 1} ---")

        # Get Loaders
        train_loader, val_loader = get_train_val_loaders(fold_idx)

        # Initialize Model
        model = get_model(pretrained=Config.pretrained, device=device)

        # Train
        # train_fold handles the training loop, validation monitoring, and checkpoint saving
        best_loss = train_fold(fold_idx, train_loader, val_loader, model)

        # OOF Inference
        # We reload the best model to ensure we are evaluating the optimal state
        print(f"Generating OOF predictions for Fold {fold_idx}...")
        probs, targets = run_oof_inference(fold_idx, val_loader, device)

        all_oof_preds.append(probs)
        all_oof_targets.append(targets)

        # Get filepaths for this fold's validation set
        # val_loader is created from df[df['fold'] == fold_idx] without shuffling
        val_fold_df = folds_df[folds_df["fold"] == fold_idx].reset_index(drop=True)
        all_oof_filepaths.extend(val_fold_df["filepath"].tolist())

        # Free memory
        del model, train_loader, val_loader
        torch.cuda.empty_cache()

    # 4. Global Validation
    global_preds = np.concatenate(all_oof_preds)
    global_targets = np.concatenate(all_oof_targets)

    # Calculate Metric
    final_metric = log_loss(global_targets, global_preds)
    print(f"\nFinal Validation Metric: {final_metric}")

    # 5. Failure Analysis
    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame(
        {"filepath": all_oof_filepaths, "prob": global_preds, "label": global_targets}
    )

    perform_failure_analysis(analysis_df)

    # 6. Submission Logic
    # Threshold from requirements
    THRESHOLD = 0.018199009307556684

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        run_inference()
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
