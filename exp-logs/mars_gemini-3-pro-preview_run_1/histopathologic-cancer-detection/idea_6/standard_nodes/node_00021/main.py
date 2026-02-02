import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.utils import set_seed, calculate_auc
from library.cross_validation import run_cv
from library.dataset import get_fold_dataloaders
from library.models import get_model
from library.inference import predict


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for Fast Baseline Execution
    # We use 7 epochs which is sufficient for convergence with pretraining
    # We use 5 folds definition but only execute Fold 0 to save time
    Config.EPOCHS = 7
    Config.N_FOLDS = 5

    # Define the specific fold to run
    target_fold = 0
    folds_to_run = [target_fold]

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print("Configuration:")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Folds to run: {folds_to_run}")
    print(f"  Models: {Config.MODEL_ARCHS}")

    # -------------------------------------------------------------------------
    # 2. Training Phase
    # -------------------------------------------------------------------------
    # Run Cross-Validation on the specified fold(s)
    # This handles training, saving checkpoints, and logging
    run_cv(folds=folds_to_run)

    # -------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\nStarting Validation and Failure Analysis...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the validation DataLoader for the trained fold
    # This ensures we evaluate on data NOT seen during training (Hold-out for this fold)
    _, val_loader = get_fold_dataloaders(target_fold, load_cached_data=True)

    # Identify trained checkpoints
    trained_models = []
    for arch in Config.MODEL_ARCHS:
        ckpt_path = os.path.join(Config.WORK_DIR, f"{arch}_fold{target_fold}_best.pth")
        if os.path.exists(ckpt_path):
            model = get_model(arch, pretrained=False)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.to(device)
            model.eval()
            trained_models.append(model)
        else:
            print(f"Warning: Checkpoint for {arch} not found.")

    if not trained_models:
        print("Error: No models available for validation.")
        return

    # Perform Inference on Validation Set
    all_preds = []
    all_targets = []

    # Store stats for failure analysis: (brightness, contrast)
    img_stats = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Ensemble Prediction (Average of models)
            batch_probs = []
            for model in trained_models:
                logits = model(images)
                probs = torch.sigmoid(logits)
                batch_probs.append(probs.cpu().numpy())

            # Average across models
            avg_probs = np.mean(np.stack(batch_probs), axis=0)

            all_preds.extend(avg_probs.flatten())
            all_targets.extend(labels.cpu().numpy().flatten())

            # Compute Image Stats for Failure Analysis
            # images is (B, C, H, W), normalized.
            # We compute stats on the tensor directly.
            # Brightness: Mean intensity
            b = images.mean(dim=(1, 2, 3)).cpu().numpy()
            # Contrast: Std dev of intensity
            c = images.std(dim=(1, 2, 3)).cpu().numpy()

            for i in range(len(b)):
                img_stats.append((b[i], c[i]))

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    img_stats = np.array(img_stats)

    # Calculate Final Metric
    final_auc = calculate_auc(y_true, y_pred)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    # Calculate error magnitude
    errors = np.abs(y_true - y_pred)
    brightness_vals = img_stats[:, 0]
    contrast_vals = img_stats[:, 1]

    # Calculate correlations
    corr_bright, _ = pearsonr(errors, brightness_vals)
    corr_contrast, _ = pearsonr(errors, contrast_vals)

    print("-" * 30)
    print("Failure Analysis Report")
    print(f"Correlation (Error vs Brightness): {corr_bright:.4f}")
    print(f"Correlation (Error vs Contrast):   {corr_contrast:.4f}")
    print("-" * 30)

    # -------------------------------------------------------------------------
    # 4. Submission
    # -------------------------------------------------------------------------
    threshold = 0.9849192531860572

    if final_auc > threshold:
        print(
            f"Metric ({final_auc:.6f}) > Threshold ({threshold:.6f}). Generating Submission..."
        )

        # Prepare config for inference
        models_config = []
        for arch in Config.MODEL_ARCHS:
            ckpt_path = os.path.join(
                Config.WORK_DIR, f"{arch}_fold{target_fold}_best.pth"
            )
            models_config.append((arch, ckpt_path))

        predict(models_config=models_config)
    else:
        print(
            f"Metric ({final_auc:.6f}) <= Threshold ({threshold:.6f}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
