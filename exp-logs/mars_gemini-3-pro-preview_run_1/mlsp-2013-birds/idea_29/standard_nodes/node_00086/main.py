import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from provided library files
from library.config import Config
from library.utils import set_seed, compute_roc_auc
from library.data import get_dataloaders, get_test_dataloader
from library.model import get_model
from library.training import run_training, validate
from library.distillation import generate_pseudo_labels

# Suppress warnings
warnings.filterwarnings("ignore")


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model errors on the validation set and correlates them with input features.
    """
    model.eval()
    all_errors = []
    img_means = []
    img_stds = []

    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Calculate Mean Absolute Error per sample across all classes
            # Shape: (Batch_Size,)
            sample_errors = torch.abs(labels - probs).mean(dim=1).cpu().numpy()
            all_errors.extend(sample_errors)

            # Calculate Image Statistics (Mean and Std per image)
            # images shape: (B, C, H, W) -> mean over (1,2,3)
            # Note: images are normalized, but relative differences still hold
            batch_means = images.mean(dim=(1, 2, 3)).cpu().numpy()
            batch_stds = images.std(dim=(1, 2, 3)).cpu().numpy()

            img_means.extend(batch_means)
            img_stds.extend(batch_stds)

    all_errors = np.array(all_errors)
    img_means = np.array(img_means)
    img_stds = np.array(img_stds)

    # Calculate Correlations
    corr_mean = np.corrcoef(all_errors, img_means)[0, 1]
    corr_std = np.corrcoef(all_errors, img_stds)[0, 1]

    print("-" * 30)
    print("Failure Analysis Report")
    print("-" * 30)
    print(f"Correlation (Error vs Image Intensity): {corr_mean:.4f}")
    print(f"Correlation (Error vs Image Contrast):  {corr_std:.4f}")
    print("-" * 30)


def generate_submission(model, device):
    """
    Generates the submission file for the test set.
    """
    print("Generating submission...")
    test_loader, rec_ids = get_test_dataloader(load_cached_data=True)
    model.eval()

    all_probs = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())

    if len(all_probs) == 0:
        print("Error: No predictions generated.")
        return

    final_probs = np.concatenate(all_probs, axis=0)  # (N_samples, 19)

    # Format submission: Id, Probability
    # Id = rec_id * 100 + species_id
    submission_rows = []

    num_classes = Config.NUM_CLASSES

    for i, rid in enumerate(rec_ids):
        probs_sample = final_probs[i]
        for species_idx in range(num_classes):
            row_id = int(rid * 100 + species_idx)
            prob = probs_sample[species_idx]
            submission_rows.append({"Id": row_id, "Probability": prob})

    df_sub = pd.DataFrame(submission_rows)

    # Ensure output directory exists
    os.makedirs("./submission", exist_ok=True)
    sub_path = "./submission/submission.csv"
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # =========================================================================
    # Stage 1: Train Teacher Ensemble
    # =========================================================================
    print("\n=== Stage 1: Training Teacher Ensemble ===")

    # Teacher 1: Linearity Bias (High Mixup)
    print("\nTraining Teacher 1 (Linearity Bias)...")
    train_loader_t1, val_loader_t1 = get_dataloaders(teacher_policy="POLICY_TEACHER_1")
    model_t1 = get_model(device=device)
    run_training(
        model=model_t1,
        train_loader=train_loader_t1,
        val_loader=val_loader_t1,
        swa_start_epoch=Config.TEACHER_SWA_START_EPOCH,
        save_path=Config.TEACHER_1_CHECKPOINT,
        mixup_alpha=Config.POLICY_TEACHER_1["mixup_alpha"],
    )

    # Teacher 2: Occlusion Robustness (High Cutout)
    print("\nTraining Teacher 2 (Occlusion Robustness)...")
    train_loader_t2, val_loader_t2 = get_dataloaders(teacher_policy="POLICY_TEACHER_2")
    model_t2 = get_model(device=device)
    run_training(
        model=model_t2,
        train_loader=train_loader_t2,
        val_loader=val_loader_t2,
        swa_start_epoch=Config.TEACHER_SWA_START_EPOCH,
        save_path=Config.TEACHER_2_CHECKPOINT,
        mixup_alpha=Config.POLICY_TEACHER_2["mixup_alpha"],
    )

    # Teacher 3: Balanced
    print("\nTraining Teacher 3 (Balanced)...")
    train_loader_t3, val_loader_t3 = get_dataloaders(teacher_policy="POLICY_BALANCED")
    model_t3 = get_model(device=device)
    run_training(
        model=model_t3,
        train_loader=train_loader_t3,
        val_loader=val_loader_t3,
        swa_start_epoch=Config.TEACHER_SWA_START_EPOCH,
        save_path=Config.TEACHER_3_CHECKPOINT,
        mixup_alpha=Config.POLICY_BALANCED["mixup_alpha"],
    )

    # =========================================================================
    # Stage 2: Pseudo-Label Generation
    # =========================================================================
    print("\n=== Stage 2: Generating Pseudo-Labels ===")
    # Clean up memory before inference
    del model_t1, model_t2, model_t3
    torch.cuda.empty_cache()

    generate_pseudo_labels()

    # =========================================================================
    # Stage 3: Train Student Model
    # =========================================================================
    print("\n=== Stage 3: Training Student Model ===")

    # Load combined dataloaders
    train_loader_s, val_loader_s = get_dataloaders(
        teacher_policy="POLICY_BALANCED", use_pseudo_labels=True
    )

    model_student = get_model(device=device)

    # Train Student
    student_swa_model = run_training(
        model=model_student,
        train_loader=train_loader_s,
        val_loader=val_loader_s,
        swa_start_epoch=Config.STUDENT_SWA_START_EPOCH,
        save_path=Config.STUDENT_CHECKPOINT,
        mixup_alpha=Config.POLICY_BALANCED["mixup_alpha"],
    )

    # =========================================================================
    # Stage 4: Validation and Analysis
    # =========================================================================
    print("\n=== Stage 4: Final Evaluation ===")

    # Validate Student
    criterion = torch.nn.BCEWithLogitsLoss()
    val_loss, val_auc = validate(student_swa_model, val_loader_s, criterion, device)

    print(f"Final Validation Metric: {val_auc}")

    # Failure Analysis
    perform_failure_analysis(student_swa_model, val_loader_s, device)

    # =========================================================================
    # Stage 5: Submission
    # =========================================================================
    THRESHOLD = 0.9594082190886809

    if val_auc > THRESHOLD:
        print(f"\nMetric ({val_auc}) > Threshold ({THRESHOLD}). Generating submission.")
        generate_submission(student_swa_model, device)
    else:
        print(f"\nMetric ({val_auc}) <= Threshold ({THRESHOLD}). Skipping submission.")


if __name__ == "__main__":
    main()
