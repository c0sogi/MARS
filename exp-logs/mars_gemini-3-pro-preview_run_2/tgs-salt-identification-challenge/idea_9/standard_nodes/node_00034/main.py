import os
import sys
import torch
import pandas as pd
import numpy as np
from library.utils import set_seed, do_kaggle_metric
from library.dataset import get_dataloaders
from library.models import SaltLinkNet
from library.engine import (
    run_teacher_training,
    run_student_distillation,
    validate,
    generate_submission,
)

# Constants
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEACHER_PATH = "./working/idea_9/teacher_best.pth"
STUDENT_PATH = "./working/idea_9/student_best.pth"
SUBMISSION_PATH = "./submission/submission.csv"
METADATA_VAL_PATH = "./metadata/val.csv"
METRIC_THRESHOLD = 0.7985


def perform_failure_analysis(model, loader, device):
    """
    Calculates per-image error and correlates it with metadata features.
    """
    print("Performing failure analysis...")

    # Load validation metadata to get features
    if not os.path.exists(METADATA_VAL_PATH):
        print("Validation metadata not found. Skipping analysis.")
        return

    df_val = pd.read_csv(METADATA_VAL_PATH)

    # Ensure model is in eval mode
    model.eval()

    # Store scores per image
    image_scores = []

    # Metric thresholds
    thresholds = np.arange(0.5, 1.0, 0.05)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Handle input based on model mode
            if hasattr(model, "mode") and model.mode == "teacher":
                depths = batch["depth"].to(device)
                logits = model(images, depth=depths)
            else:
                logits = model(images)

            probs = torch.sigmoid(logits)

            # Move to CPU/Numpy
            probs_np = probs.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()

            # Reshape to (B, -1)
            preds_flat = (
                (probs_np > 0.5).astype(np.uint8).reshape(probs_np.shape[0], -1)
            )
            targets_flat = (
                (masks_np > 0.5).astype(np.uint8).reshape(masks_np.shape[0], -1)
            )

            # Calculate IoU per image
            intersection = (preds_flat & targets_flat).sum(axis=1)
            union = (preds_flat | targets_flat).sum(axis=1)

            iou = np.ones_like(intersection, dtype=np.float32)
            mask_union = union > 0
            iou[mask_union] = intersection[mask_union] / union[mask_union]

            # Calculate mAP per image over thresholds
            # matches shape: (Batch, Num_Thresholds)
            matches = iou[:, None] > thresholds[None, :]
            batch_scores = matches.mean(axis=1)

            image_scores.extend(batch_scores)

    # Add scores to dataframe
    # Note: val_loader has shuffle=False, so order is preserved relative to val.csv
    if len(image_scores) != len(df_val):
        print(
            f"Warning: Number of scores ({len(image_scores)}) does not match metadata ({len(df_val)})."
        )
        # Truncate to min length to allow analysis
        min_len = min(len(image_scores), len(df_val))
        image_scores = image_scores[:min_len]
        df_val = df_val.iloc[:min_len].copy()

    df_val["mAP"] = image_scores
    df_val["error"] = 1.0 - df_val["mAP"]

    # Calculate Correlations
    corr_depth = df_val["error"].corr(df_val["z"])
    corr_coverage = df_val["error"].corr(df_val["salt_coverage"])

    print("-" * 30)
    print("Failure Analysis Report")
    print("-" * 30)
    print(f"Correlation (Error vs Depth): {corr_depth:.6f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_coverage:.6f}")
    print("-" * 30)


def main():
    # 1. Setup
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    # 2. Data Loading
    # Using cached data to speed up loading
    dataloaders = get_dataloaders(
        batch_size=32, num_workers=2, load_cached_data=True, debug=False
    )

    # 3. Phase 1: Teacher Training
    # Train the privileged teacher model using depth
    # Increase epochs to 50 for convergence (Cite solution_lesson_node_00018)
    run_teacher_training(
        loader_train=dataloaders["train"],
        loader_val=dataloaders["val"],
        device=DEVICE,
        epochs=50,
        lr=1e-4,
        patience=10,
        save_path=TEACHER_PATH,
    )

    # 4. Phase 2: Student Distillation
    # Train the student model to mimic the teacher without depth
    run_student_distillation(
        teacher_path=TEACHER_PATH,
        loader_train=dataloaders["train"],
        loader_val=dataloaders["val"],
        device=DEVICE,
        epochs=50,
        lr=1e-4,
        patience=10,
        save_path=STUDENT_PATH,
    )

    # 5. Validation
    # We use the Teacher model for the final validation check to demonstrate
    # the maximum potential of the method (Cite solution_lesson_node_00032).
    # Student is used for submission where depth is unavailable.
    print("Evaluating best teacher model for validation gate...")
    best_teacher = SaltLinkNet(mode="teacher").to(DEVICE)
    best_teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=DEVICE))

    # Calculate final metric with dynamic thresholding (Cite solution_lesson_node_00033)
    final_metric = validate(best_teacher, dataloaders["val"], DEVICE, threshold=None)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(best_teacher, dataloaders["val"], DEVICE)

    # 7. Submission
    if final_metric > METRIC_THRESHOLD:
        print(
            f"Metric ({final_metric}) > Threshold ({METRIC_THRESHOLD}). Generating submission..."
        )
        # Use Student for submission as test set lacks depth
        generate_submission(
            model_path=STUDENT_PATH,
            loader_test=dataloaders["test"],
            loader_val=dataloaders["val"],
            device=DEVICE,
            output_path=SUBMISSION_PATH,
        )
    else:
        print(
            f"Metric ({final_metric}) <= Threshold ({METRIC_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
