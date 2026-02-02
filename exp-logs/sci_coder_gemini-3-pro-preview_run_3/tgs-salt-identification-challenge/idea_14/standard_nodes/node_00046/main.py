import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_iou_batch, rle_encode
from library.dataset import get_loaders, get_test_loader, SaltDataset, get_transforms
from library.model import SaltUNetPlusPlus
from library.engine import SaltEngine
from library.inference import predict_with_tta, optimize_threshold, generate_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("Initializing Fast Baseline Run...")
    seed_everything(Config.SEED)

    # Configuration
    FOLD_TO_RUN = 0
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Supervised Training (Fold 0)
    # -------------------------------------------------------------------------
    print(f"\n--- Supervised Training (Fold {FOLD_TO_RUN}) ---")

    # Get Loaders
    train_loader, val_loader = get_loaders(fold=FOLD_TO_RUN, load_cached_data=True)

    # Initialize Model
    model = SaltUNetPlusPlus(
        encoder_name=Config.ENCODER,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=Config.CHANNELS,
        classes=1,
    ).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Engine
    engine = SaltEngine(model, device, optimizer, scheduler)

    # Train
    # We remove SSL (Phase 2) as it causes instability in mixed precision (Cite solution_lesson_node_00045)
    # We extend training to 50 epochs to allow Lovasz loss to converge (Cite solution_lesson_node_00020)
    best_model_path = os.path.join(
        Config.CHECKPOINT_DIR, f"fold_{FOLD_TO_RUN}_best.pth"
    )
    engine.fit(
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        save_path=best_model_path,
        phase2=False,
        patience=10,
    )

    # Load Best Weights
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Load test loader for submission
    test_loader = get_test_loader(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Final Validation & Failure Analysis ---")

    # Run Inference on Validation Set
    val_results = predict_with_tta(model, val_loader, device)
    val_preds = val_results["preds"]
    val_targets = val_results["targets"]

    # Calculate Final Metric
    final_metric = calculate_iou_batch(val_preds, val_targets, threshold=0.5)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis
    print("Performing Failure Analysis...")

    # Calculate per-image mAP (Error = 1 - mAP)
    errors = []
    # We need to replicate the metric calculation logic per image
    iou_thresholds = np.arange(0.5, 0.96, 0.05)

    for i in range(len(val_preds)):
        p = (val_preds[i] > 0.5).astype(np.uint8)
        t = (val_targets[i] > 0.5).astype(np.uint8)

        intersection = np.sum(p & t)
        union = np.sum(p | t)
        iou = 1.0 if union == 0 else intersection / union

        matches = iou > iou_thresholds
        score = np.mean(matches)
        errors.append(1.0 - score)

    errors = np.array(errors)

    # Extract Metadata Features
    val_ds = val_loader.dataset
    depths = val_ds.depths

    # Calculate Salt Coverage (Target)
    coverages = np.array([np.mean(t) for t in val_targets])

    # Calculate Image Intensity
    intensities = np.array([np.mean(img) for img in val_ds.images]) / 255.0

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "depth": depths,
            "coverage": coverages,
            "intensity": intensities,
        }
    )

    # Compute Correlations
    corr_depth = analysis_df["error"].corr(analysis_df["depth"])
    corr_cov = analysis_df["error"].corr(analysis_df["coverage"])
    corr_int = analysis_df["error"].corr(analysis_df["intensity"])

    print(f"Error Correlation with Depth: {corr_depth:.4f}")
    print(f"Error Correlation with Salt Coverage: {corr_cov:.4f}")
    print(f"Error Correlation with Image Intensity: {corr_int:.4f}")

    # -------------------------------------------------------------------------
    # 4. Submission
    # -------------------------------------------------------------------------
    if final_metric > 0.827:
        print("\n--- Generating Submission ---")

        # Optimize Threshold
        best_threshold = optimize_threshold(val_preds, val_targets)

        # Predict on Test Set
        test_results_final = predict_with_tta(model, test_loader, device)

        # Generate CSV
        generate_submission(
            test_results_final["preds"],
            test_results_final["ids"],
            best_threshold,
            Config.SUBMISSION_PATH,
        )
    else:
        print(f"\nMetric {final_metric:.4f} <= 0.827. Skipping submission generation.")


if __name__ == "__main__":
    main()
