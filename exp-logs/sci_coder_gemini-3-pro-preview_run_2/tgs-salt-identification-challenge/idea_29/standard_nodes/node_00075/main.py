import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Add library path if needed
sys.path.append(".")

from library.config import Config, seed_everything
from library.pipeline import (
    run_stage1_teacher,
    run_stage2_student_ensemble,
    run_stage3_self_training,
)
from library.models import StudentLinkNet
from library.dataset import get_loaders
from library.utils import unpad_image, calc_map
from library.training import generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)

    # 2. Configure for Fast Baseline
    # Adjust epochs to ensure completion within 2 hours while maintaining performance.
    # 20 epochs per stage is sufficient for the dataset size (2400 train images).
    Config.NUM_EPOCHS_TEACHER = 20
    Config.NUM_EPOCHS_STUDENT = 20
    Config.NUM_EPOCHS_FINAL = 20

    # Ensure working directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print("--- Starting Privileged Multi-Task Ensemble Distillation Pipeline ---")

    # 3. Execute Pipeline Stages

    # Stage 1: Privileged Teacher (Image + Depth)
    # Trains on labeled data using depth injection
    teacher_model_path = run_stage1_teacher(debug=False)

    # Stage 2: Multi-Task Student Ensemble (Image Only)
    # Distills knowledge from Teacher and learns auxiliary depth regression
    student_model_paths = run_stage2_student_ensemble(teacher_model_path, debug=False)

    # Stage 3: Self-Training (Noisy Student)
    # Generates pseudo-labels on Test set and retrains final model
    # This function automatically generates submission.csv if successful
    run_stage3_self_training(student_model_paths, debug=False)

    # 4. Final Validation and Failure Analysis
    print("\n--- Performing Final Validation & Failure Analysis ---")

    # Determine which model to evaluate
    # Prefer Stage 3 model, fallback to best Stage 2 model if Stage 3 didn't run
    stage3_model_path = os.path.join(Config.CACHE_DIR, "best_model_stage3.pth")

    final_model_path = None
    if os.path.exists(stage3_model_path):
        final_model_path = stage3_model_path
        print("Using Stage 3 Best Model.")
    elif student_model_paths:
        final_model_path = student_model_paths[0]  # Take the first valid fold
        print("Stage 3 model not found. Falling back to Stage 2 Model (Fold 0).")
    else:
        print("No valid models trained. Exiting.")
        return

    # Load Model
    model = StudentLinkNet(num_classes=1).to(Config.DEVICE)
    model.load_state_dict(torch.load(final_model_path, map_location=Config.DEVICE))
    model.eval()

    # Load Validation Data
    # get_loaders returns (train, val, test)
    _, val_loader, _ = get_loaders(load_cached_data=True)

    # Load Best Threshold (saved during training)
    thresh_path = os.path.join(Config.CACHE_DIR, "best_threshold.txt")
    threshold = 0.5
    if os.path.exists(thresh_path):
        with open(thresh_path, "r") as f:
            try:
                threshold = float(f.read().strip())
            except:
                pass
    print(f"Using optimized threshold: {threshold}")

    # Inference on Validation Set
    all_preds = []
    all_masks = []
    all_depths = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(Config.DEVICE).float()
            masks = batch["mask"].numpy()
            depths = batch["depth"].numpy()  # Normalized depths

            # Student model only needs images
            outputs = model(images)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(len(probs)):
                # Unpad to original size (101x101)
                p = unpad_image(probs[i].squeeze(0))
                m = unpad_image(masks[i].squeeze(0))

                all_preds.append(p)
                all_masks.append(m)
                all_depths.append(depths[i][0])

    all_preds = np.array(all_preds)
    all_masks = np.array(all_masks)
    all_depths = np.array(all_depths)

    # 5. Compute Metrics

    # Apply threshold to get binary masks
    binary_preds = (all_preds > threshold).astype(np.uint8)

    # Calculate mAP (Metric for Competition)
    final_map = calc_map(binary_preds, all_masks)
    print(f"Final Validation Metric: {final_map:.10f}")

    # 6. Failure Analysis
    # Calculate IoU per image to correlate with Depth
    intersection = np.sum(binary_preds & all_masks, axis=(1, 2))
    union = np.sum(binary_preds | all_masks, axis=(1, 2))

    # Handle division by zero (empty union means both empty -> IoU=1)
    ious = np.ones_like(intersection, dtype=np.float32)
    valid_mask = union > 0
    ious[valid_mask] = intersection[valid_mask] / union[valid_mask]

    errors = 1.0 - ious

    # Correlation (Pearson) between Error and Depth
    if np.std(errors) > 0 and np.std(all_depths) > 0:
        correlation = np.corrcoef(errors, all_depths)[0, 1]
    else:
        correlation = 0.0

    print(f"Failure Analysis - Correlation (Error vs Depth): {correlation:.10f}")

    # 7. Submission Logic
    # Only submit if metric > 0.7985
    if final_map > 0.7985:
        print("Validation metric meets threshold. Submission preserved.")
        # Ensure submission exists (generated by Stage 3). If not, generate it now.
        if not os.path.exists(Config.SUBMISSION_PATH):
            print("Generating submission file manually...")
            _, _, test_loader = get_loaders(load_cached_data=True)
            generate_submission(
                model, test_loader, Config.DEVICE, Config.SUBMISSION_PATH
            )
    else:
        print(f"Validation metric {final_map} <= 0.7985. Discarding submission.")
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
