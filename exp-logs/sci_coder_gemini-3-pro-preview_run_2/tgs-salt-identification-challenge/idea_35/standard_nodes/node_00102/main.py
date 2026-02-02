"""
Runfile for Salt Segmentation Task.
Implements the Marginalized-Scan Multi-Task Distillation strategy.
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import Library Components
from library.config import Config
from library.utils import set_seed, do_kaggle_metric, unpad_image_128
from library.dataset import prepare_data, SaltDataset, get_transforms
from library.models import SaltNet
from library.stages import (
    run_stage_1_teacher_ensemble,
    run_stage_2_marginalization,
    run_stage_3_student_distillation,
)
from library.engine import validate, optimize_threshold, generate_submission


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on validation set.
    Calculates correlation between Error (1-mAP) and Depth/SaltCoverage.
    """
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    model.eval()
    errors = []
    depths = []
    coverages = []

    with torch.no_grad():
        for data in val_loader:
            images = data["image"].to(device)
            masks = data["mask"].to(device)
            # Get depths (scaled) for correlation
            batch_depths = data["depth"].cpu().numpy().flatten()

            # Student model inference (Image only)
            if hasattr(model, "aux_head") and model.aux_head:
                logits, _ = model(images)
            else:
                logits = model(images)

            probs = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(probs_np.shape[0]):
                # Unpad to original size for accurate metric calculation
                p = probs_np[i].transpose(1, 2, 0)
                m = masks_np[i].transpose(1, 2, 0)

                p_orig = unpad_image_128(p)
                m_orig = unpad_image_128(m)

                # Calculate metric for this single image
                score = do_kaggle_metric(p_orig[None, ...], m_orig[None, ...])
                error = 1.0 - score

                # Calculate Salt Coverage (Salt pixels / Total pixels)
                salt_pixels = np.sum(m_orig)
                coverage = salt_pixels / (101 * 101)

                errors.append(error)
                depths.append(batch_depths[i])
                coverages.append(coverage)

    errors = np.array(errors)
    depths = np.array(depths)
    coverages = np.array(coverages)

    # Correlation with Depth
    if len(np.unique(depths)) > 1:
        corr_depth, _ = pearsonr(errors, depths)
        print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    else:
        print("Correlation (Error vs Depth): Undefined (Constant Depth)")

    # Correlation with Coverage
    if len(np.unique(coverages)) > 1:
        corr_cov, _ = pearsonr(errors, coverages)
        print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")
    else:
        print("Correlation (Error vs Salt Coverage): Undefined")


def main():
    # 1. Configuration for Fast Baseline
    # We override Config attributes to ensure the pipeline completes within the time limit.
    # 20 epochs is sufficient for this dataset size to get reasonable results.
    # 3 folds provides a balance between ensemble robustness and runtime.
    Config.EPOCHS_STAGE1 = 20
    Config.EPOCHS_STAGE3 = 20
    Config.N_FOLDS = 3

    # Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Configuration:")
    print(f"  Device: {device}")
    print(f"  Stage 1 Epochs: {Config.EPOCHS_STAGE1}")
    print(f"  Stage 3 Epochs: {Config.EPOCHS_STAGE3}")
    print(f"  Folds: {Config.N_FOLDS}")

    # 2. Data Loading
    # Load cached data if available to save processing time
    data_containers = prepare_data(load_cached_data=True)

    # 3. Pipeline Execution

    # Stage 1: Teacher Ensemble
    teacher_paths = run_stage_1_teacher_ensemble(data_containers, device)

    if not teacher_paths:
        print(
            "Pipeline aborted: No teacher models passed the gating threshold (0.75 mAP)."
        )
        return

    # Stage 2: Marginalization
    soft_masks = run_stage_2_marginalization(teacher_paths, data_containers, device)

    # Stage 3: Student Distillation
    student_model_path = run_stage_3_student_distillation(
        soft_masks, data_containers, device
    )

    # 4. Evaluation
    print("\n" + "=" * 40)
    print("FINAL EVALUATION")
    print("=" * 40)

    # Load Best Student Model
    model = SaltNet(use_depth=False, aux_head=True, pretrained=False).to(device)
    model.load_state_dict(torch.load(student_model_path, map_location=device))
    model.eval()

    # Validation Loader
    val_ds = SaltDataset(
        data_containers["val"]["images"],
        masks=data_containers["val"]["masks"],
        depths=data_containers["val"]["depths"],
        ids=data_containers["val"]["ids"],
        transform=get_transforms("val"),
        mode="val",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Calculate Final Metric
    final_metric = validate(model, val_loader, device, is_student=True)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 5. Submission
    SUBMISSION_THRESHOLD = 0.7985

    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"\nMetric ({final_metric:.4f}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        # Optimize Threshold
        best_threshold = optimize_threshold(model, val_loader, device)

        # Test Loader
        test_ds = SaltDataset(
            data_containers["test"]["images"],
            ids=data_containers["test"]["ids"],
            transform=get_transforms("test"),
            mode="test",
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Generate Submission
        generate_submission(model, test_loader, device, threshold=best_threshold)
    else:
        print(
            f"\nMetric ({final_metric:.4f}) did not exceed threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
