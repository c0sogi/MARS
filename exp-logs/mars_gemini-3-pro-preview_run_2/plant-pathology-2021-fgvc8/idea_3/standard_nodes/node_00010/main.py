import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import CFG
from library.utils import seed_everything, get_score
from library.data import get_loaders
from library.model import AppleDiseaseModel
from library.engine import fit_model
from library.inference import predict_tta, soft_voting_ensemble, generate_submission


def analyze_failures(val_loader, y_true, y_pred_probs):
    """
    Performs failure analysis by correlating error magnitude with image metadata.
    """
    print("\n=== Failure Analysis ===")

    # Calculate error magnitude (mean absolute difference)
    # y_true is binary (N, C), y_pred_probs is float (N, C)
    error_magnitude = np.mean(np.abs(y_true - y_pred_probs), axis=1)

    # Get metadata from the validation dataframe
    val_df = val_loader.dataset.df

    meta_stats = []
    print("Extracting metadata for failure analysis...")

    for idx, row in val_df.iterrows():
        full_path = os.path.join(CFG.input_root, row["file_path"])

        # Get file size
        try:
            f_size = os.path.getsize(full_path)
        except OSError:
            f_size = 0

        # Get dimensions (read image)
        # We read the image to get actual dimensions, though it might be slow for large sets.
        # Given the constraints, we'll try to do this efficiently.
        img = cv2.imread(full_path)
        if img is not None:
            h, w, _ = img.shape
        else:
            h, w = 0, 0

        meta_stats.append(
            {
                "file_size": f_size,
                "width": w,
                "height": h,
                "error": error_magnitude[idx],
            }
        )

    meta_df = pd.DataFrame(meta_stats)

    # Calculate correlations
    if not meta_df.empty:
        correlations = meta_df.corr()["error"].drop("error")
        print("\nCorrelation between Error Magnitude and Input Features:")
        print(correlations)
    else:
        print("Could not extract metadata for analysis.")


def main():
    # 1. Setup
    seed_everything(CFG.seed)

    # Override CFG for fast baseline execution
    CFG.epochs = 5  # Limit epochs to ensure completion within 2 hours
    # Ensure we use the full dataset for the best chance of meeting the threshold
    CFG.debug = False

    print(f"Configuration:")
    print(f"  Device: {CFG.device}")
    print(f"  Epochs: {CFG.epochs}")
    print(f"  Backbones: {CFG.backbones}")

    # 2. Data Loading
    print("\nLoading Data...")
    train_loader, val_loader, test_loader = get_loaders()

    # 3. Training Loop
    trained_model_configs = []

    for backbone in CFG.backbones:
        print(f"\n{'='*40}")
        print(f"Training Backbone: {backbone}")
        print(f"{'='*40}")

        # Initialize Model
        model = AppleDiseaseModel(model_name=backbone, pretrained=True)
        model.to(CFG.device)

        # Optimizer & Scheduler
        optimizer = AdamW(
            model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, eps=CFG.eps
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr)

        # Train
        best_f1, save_path = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=CFG.device,
            epochs=CFG.epochs,
            model_name=backbone.split(".")[0],  # Simplify name
            patience=5,  # Minimal early stopping for short run
        )

        trained_model_configs.append((backbone, save_path))

        # Cleanup
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    # 4. Validation & Failure Analysis
    print(f"\n{'='*40}")
    print("Ensemble Validation")
    print(f"{'='*40}")

    val_probs_list = []

    # Get Ground Truth
    # We need to iterate the loader to get targets in the same order
    y_true_list = []
    for _, targets in val_loader:
        y_true_list.append(targets.numpy())
    y_true = np.concatenate(y_true_list, axis=0)

    # Predict with each model
    for backbone, weight_path in trained_model_configs:
        print(f"Predicting validation set with {backbone}...")
        model = AppleDiseaseModel(model_name=backbone, pretrained=False)
        model.load_state_dict(torch.load(weight_path, map_location=CFG.device))
        model.to(CFG.device)

        probs = predict_tta(model, val_loader, CFG.device)
        val_probs_list.append(probs)

        del model
        torch.cuda.empty_cache()

    # Ensemble
    avg_val_probs = np.mean(val_probs_list, axis=0)

    # Calculate Metric
    # Threshold 0.5 is used in get_score default
    final_metric = get_score(y_true, avg_val_probs, threshold=0.5)

    print(f"\nFinal Validation Metric: {final_metric}")

    # Failure Analysis
    analyze_failures(val_loader, y_true, avg_val_probs)

    # 5. Submission
    THRESHOLD = 0.9228752356223593

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(trained_model_configs, test_loader, CFG.device)
    else:
        print(
            f"\nValidation metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
