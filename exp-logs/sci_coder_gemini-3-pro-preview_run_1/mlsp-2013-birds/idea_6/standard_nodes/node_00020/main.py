import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import set_seed, Logger, save_checkpoint, calculate_roc_auc
from library.dataset import get_data_loaders
from library.model import BirdResNet
from library.engine import train_one_epoch, validate, inference


def main():
    # --- 1. Setup & Configuration ---
    set_seed(Config.SEED)

    # Initialize Logger
    logger = Logger(os.path.join(Config.WORKING_DIR, "training_log.txt"))
    logger.log("Starting Spectral-Dynamic ResNet-34 with Self-Distillation Pipeline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}")

    # --- 2. Data Loading (Stage 1) ---
    logger.log("\n--- Loading Data for Teacher Training ---")
    # Load initial loaders (Train only on labeled data)
    train_loader_teacher, val_loader, test_loader = get_data_loaders(
        Config, pseudo_labels=None, load_cached_data=True
    )

    # --- 3. Stage 1: Teacher Training ---
    logger.log("\n--- Stage 1: Teacher Training ---")

    teacher_model = BirdResNet(num_classes=Config.NUM_CLASSES, pretrained=True).to(
        device
    )

    optimizer_teacher = optim.AdamW(
        teacher_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler_teacher = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_teacher, T_max=Config.TEACHER_EPOCHS
    )

    best_teacher_auc = 0.0

    for epoch in range(1, Config.TEACHER_EPOCHS + 1):
        train_loss = train_one_epoch(
            teacher_model, train_loader_teacher, optimizer_teacher, device, Config
        )
        val_loss, val_auc = validate(teacher_model, val_loader, device)
        scheduler_teacher.step()

        logger.log_metrics(epoch, train_loss, val_loss, val_auc)

        if val_auc > best_teacher_auc:
            best_teacher_auc = val_auc
            save_checkpoint(teacher_model, Config.TEACHER_MODEL_PATH)
            logger.log(f"  -> New Best Teacher Model Saved! AUC: {val_auc:.4f}")

    logger.log(f"Best Teacher AUC: {best_teacher_auc:.4f}")

    # --- 4. Pseudo-Labeling ---
    logger.log("\n--- Generating Pseudo-Labels for Student Training ---")

    # Load best teacher
    teacher_model.load_state_dict(
        torch.load(Config.TEACHER_MODEL_PATH, map_location=device)
    )

    # Inference on Test Set
    test_ids, test_probs = inference(teacher_model, test_loader, device)

    # Map IDs to probabilities for correct ordering in dataset merging
    # The get_data_loaders function expects an array matching the test set order or a dict.
    # We'll pass a dict to be safe as implemented in dataset.py
    pseudo_labels_dict = {rid: probs for rid, probs in zip(test_ids, test_probs)}

    # --- 5. Stage 2: Student Training ---
    logger.log("\n--- Stage 2: Student Training (Self-Distillation) ---")

    # Reload loaders with pseudo-labels
    # This merges Train and Test sets for the training loader
    train_loader_student, _, _ = get_data_loaders(
        Config, pseudo_labels=pseudo_labels_dict, load_cached_data=True
    )

    student_model = BirdResNet(num_classes=Config.NUM_CLASSES, pretrained=True).to(
        device
    )

    optimizer_student = optim.AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler_student = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_student, T_max=Config.STUDENT_EPOCHS
    )

    best_student_auc = 0.0

    for epoch in range(1, Config.STUDENT_EPOCHS + 1):
        train_loss = train_one_epoch(
            student_model, train_loader_student, optimizer_student, device, Config
        )
        # Validate on the original validation set (Fold 0 hold-out)
        val_loss, val_auc = validate(student_model, val_loader, device)
        scheduler_student.step()

        logger.log_metrics(epoch, train_loss, val_loss, val_auc)

        if val_auc > best_student_auc:
            best_student_auc = val_auc
            save_checkpoint(student_model, Config.STUDENT_MODEL_PATH)
            logger.log(f"  -> New Best Student Model Saved! AUC: {val_auc:.4f}")

    # --- 6. Final Evaluation & Failure Analysis ---
    logger.log("\n--- Final Evaluation & Failure Analysis ---")

    # Load best student
    student_model.load_state_dict(
        torch.load(Config.STUDENT_MODEL_PATH, map_location=device)
    )

    # Final Validation
    _, final_val_auc = validate(student_model, val_loader, device)
    print(f"Final Validation Metric: {final_val_auc}")

    # Failure Analysis
    student_model.eval()
    val_errors = []
    val_img_means = []
    val_img_stds = []
    val_label_counts = []

    criterion_none = nn.BCEWithLogitsLoss(reduction="none")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = student_model(images)
            probs = torch.sigmoid(logits)

            # Calculate Mean Absolute Error per sample
            # shape: (B, C) -> mean over C -> (B,)
            mae = torch.abs(probs - labels).mean(dim=1).cpu().numpy()
            val_errors.extend(mae)

            # Image stats (Channel 0 is intensity)
            # images shape: (B, 3, H, W)
            img_mean = images[:, 0, :, :].mean(dim=(1, 2)).cpu().numpy()
            img_std = images[:, 0, :, :].std(dim=(1, 2)).cpu().numpy()

            val_img_means.extend(img_mean)
            val_img_stds.extend(img_std)

            # Label counts
            l_counts = labels.sum(dim=1).cpu().numpy()
            val_label_counts.extend(l_counts)

    val_errors = np.array(val_errors)
    val_img_means = np.array(val_img_means)
    val_img_stds = np.array(val_img_stds)
    val_label_counts = np.array(val_label_counts)

    # Correlations
    corr_mean, _ = pearsonr(val_errors, val_img_means)
    corr_std, _ = pearsonr(val_errors, val_img_stds)
    corr_count, _ = pearsonr(val_errors, val_label_counts)

    print("\nFailure Analysis Correlations (Error Magnitude vs Feature):")
    print(f"  Signal Intensity (Mean): {corr_mean:.4f}")
    print(f"  Signal Contrast (Std):   {corr_std:.4f}")
    print(f"  Label Complexity (Count): {corr_count:.4f}")

    # --- 7. Submission ---
    threshold = 0.9255537489325414
    if final_val_auc > threshold:
        logger.log(
            f"\nValidation metric {final_val_auc} > {threshold}. Generating submission..."
        )

        # Inference on Test Set with Best Student
        test_ids, test_probs = inference(student_model, test_loader, device)

        # Format Submission
        submission_rows = []
        for i in range(len(test_ids)):
            rec_id = int(test_ids[i])
            probs = test_probs[i]
            for species_idx in range(len(probs)):
                row_id = rec_id * 100 + species_idx
                prob = probs[species_idx]
                submission_rows.append({"Id": row_id, "Probability": prob})

        df_sub = pd.DataFrame(submission_rows)

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.log(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        logger.log(
            f"\nValidation metric {final_val_auc} <= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
