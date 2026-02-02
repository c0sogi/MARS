import sys
import os
import numpy as np
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, sanitize_pseudo_labels
from library.model import BirdResNet
from library.data import get_dataloaders
from library.training import run_swa_training, validate, predict, generate_submission


def main():
    # 1. Configuration
    # We use debug=False because the dataset is small (208 samples) and
    # full training is required to meet the high AUC threshold.
    # Training time is negligible (<10 mins) even with full epochs.
    cfg = Config(debug=False)
    set_seed(cfg.SEED)

    print("========================================")
    print("   Idea 11: High-Fidelity Distillation  ")
    print("========================================")

    # ---------------------------------------------------------
    # Stage 1: Train Teachers
    # ---------------------------------------------------------
    print("\n[Stage 1] Training Teacher Ensemble...")
    teacher_preds_list = []

    # Get loaders for teacher training (Train on Fold 0, Val on Fold 0 split)
    teacher_loaders = get_dataloaders(cfg, stage="teacher")

    # Get loader for inference (Test set Fold 1) to generate pseudo-labels
    inference_loaders = get_dataloaders(cfg, stage="inference")
    test_loader = inference_loaders["test"]

    for i in range(cfg.NUM_TEACHERS):
        print(f"\nTraining Teacher {i+1}/{cfg.NUM_TEACHERS}")

        # Initialize fresh model
        model = BirdResNet(num_classes=cfg.NUM_CLASSES, pretrained=cfg.PRETRAINED)

        # Define save path
        save_path = cfg.TEACHER_CHECKPOINT_TEMPLATE.format(i)

        # Train with SWA
        trained_teacher = run_swa_training(
            cfg,
            model,
            teacher_loaders["train"],
            teacher_loaders["val"],
            epochs=cfg.TEACHER_EPOCHS,
            swa_start_epoch=cfg.TEACHER_SWA_START_EPOCH,
            save_path=save_path,
        )

        # Generate predictions for pseudo-labeling
        print(f"Generating predictions from Teacher {i+1}...")
        preds = predict(test_loader, trained_teacher, cfg.DEVICE)
        teacher_preds_list.append(preds)

        # Cleanup
        del model, trained_teacher
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Stage 2: Pseudo-Label Generation
    # ---------------------------------------------------------
    print("\n[Stage 2] Generating and Sanitizing Pseudo-Labels...")

    # Stack predictions: (Num_Teachers, N_Samples, N_Classes)
    teacher_preds_stack = np.array(teacher_preds_list)

    # Average predictions
    avg_pseudo_labels = np.mean(teacher_preds_stack, axis=0)

    # Sanitize
    try:
        sanitized_labels = sanitize_pseudo_labels(avg_pseudo_labels)
        print("Pseudo-labels sanitized successfully.")
    except AssertionError as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Stage 3: Train Student
    # ---------------------------------------------------------
    print("\n[Stage 3] Training Student with SWA...")

    # Get loaders for student (Combined Train+Test, Val on Fold 0 split)
    student_loaders = get_dataloaders(
        cfg, stage="student", pseudo_labels=sanitized_labels
    )

    # Initialize Student Model
    student_model = BirdResNet(num_classes=cfg.NUM_CLASSES, pretrained=cfg.PRETRAINED)

    # Train Student
    final_student = run_swa_training(
        cfg,
        student_model,
        student_loaders["train"],
        student_loaders["val"],
        epochs=cfg.STUDENT_EPOCHS,
        swa_start_epoch=cfg.STUDENT_SWA_START_EPOCH,
        save_path=cfg.STUDENT_CHECKPOINT,
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    print("\n[Evaluation] Validating Final Student Model...")

    # Validate on the hold-out validation set
    val_loader = student_loaders["val"]
    val_loss, val_auc = validate(
        val_loader, final_student, torch.nn.BCEWithLogitsLoss(), cfg.DEVICE
    )

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # ---------------------------------------------------------
    # Failure Analysis
    # ---------------------------------------------------------
    print("\n[Analysis] Performing Failure Analysis...")

    final_student.eval()
    all_targets = []
    all_preds = []
    all_img_means = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(cfg.DEVICE)

            # Forward pass
            outputs = final_student(images)
            preds = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

            # Compute image mean intensity (B, 3, H, W) -> (B,)
            # Averaging over channels (1), height (2), width (3)
            batch_means = images.mean(dim=(1, 2, 3)).cpu().numpy()
            all_img_means.append(batch_means)

    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    all_img_means = np.concatenate(all_img_means)

    # Compute Mean Absolute Error per sample (averaged over classes)
    # Shape: (N_samples,)
    mae_per_sample = np.mean(np.abs(all_targets - all_preds), axis=1)

    # Compute Correlation
    if len(mae_per_sample) > 1:
        # np.corrcoef returns correlation matrix
        corr_matrix = np.corrcoef(mae_per_sample, all_img_means)
        correlation = corr_matrix[0, 1]
        print(
            f"Correlation between Error Magnitude and Image Intensity: {correlation:.10f}"
        )
    else:
        print("Insufficient samples for correlation analysis.")

    # ---------------------------------------------------------
    # Submission
    # ---------------------------------------------------------
    threshold = 0.9433543480067271

    if val_auc > threshold:
        print(
            f"\nMetric ({val_auc}) > Threshold ({threshold}). Generating submission..."
        )
        generate_submission(cfg, final_student, test_loader, cfg.SUBMISSION_PATH)
    else:
        print(f"\nMetric ({val_auc}) <= Threshold ({threshold}). Submission skipped.")

    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()
