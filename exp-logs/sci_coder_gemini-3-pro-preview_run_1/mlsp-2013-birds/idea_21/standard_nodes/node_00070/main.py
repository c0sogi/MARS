import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR

# Import from provided library files
from library.config import Config
from library.utils import set_seed, get_logger, save_checkpoint, compute_roc_auc
from library.dataset import get_dataloaders
from library.model import ManifoldMixupResNet
from library.engine import train_one_epoch, validate, generate_predictions, update_bn


def train_model_stage(
    config, model_name, train_loader, val_loader, device, logger, use_mixup=True
):
    """
    Generic training loop handling standard training + SWA.
    """
    logger.info(f"Starting training for {model_name}...")

    # Initialize Model
    model = ManifoldMixupResNet(
        num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED
    )
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Standard Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    # SWA Setup
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=config.SWA_LR)

    best_auc = 0.0

    for epoch in range(config.epochs):
        # Determine if we are in SWA phase
        is_swa_phase = epoch >= config.swa_start_epoch

        # Train one epoch
        avg_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            mixup_active=use_mixup,
            alpha=config.MIXUP_ALPHA,
        )

        # Update SWA or Standard Scheduler
        if is_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Validation (using base model for monitoring, though SWA is final target)
        # We only validate periodically to save time, or every epoch if dataset is small
        if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
            val_loss, val_auc = validate(model, val_loader, device)
            if val_auc > best_auc:
                best_auc = val_auc
            # logger.info(f"Epoch {epoch+1}/{config.epochs} - Loss: {avg_loss:.4f} - Val AUC: {val_auc:.4f}")

    # End of training: Update BatchNorm for SWA model
    logger.info(f"Updating SWA BatchNorm statistics for {model_name}...")
    update_bn(train_loader, swa_model, device)

    # Validate Final SWA Model
    swa_loss, swa_auc = validate(swa_model, val_loader, device)
    logger.info(f"{model_name} Final SWA AUC: {swa_auc:.4f}")

    # Save SWA Model
    save_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_swa.pth")
    torch.save(swa_model.module.state_dict(), save_path)

    return swa_model, save_path


def main():
    # 1. Configuration & Setup
    config = Config(debug=False)  # Ensure full run
    set_seed(config.SEED)
    device = config.DEVICE
    logger = get_logger(os.path.join(config.LOG_DIR, "run.log"))

    logger.info("Configuration initialized.")
    logger.info(f"Device: {device}")

    # 2. Data Loading (Initial)
    # Load initial dataloaders for Teacher training
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # =========================================================================
    # Stage 1: Teacher Ensemble Training
    # =========================================================================
    teacher_paths = []
    teacher_models = []

    for i in range(config.NUM_TEACHERS):
        teacher_name = f"teacher_{i}"
        model, path = train_model_stage(
            config,
            teacher_name,
            train_loader,
            val_loader,
            device,
            logger,
            use_mixup=True,
        )
        teacher_models.append(model)
        teacher_paths.append(path)

    # =========================================================================
    # Stage 2: Pseudo-Label Generation
    # =========================================================================
    logger.info("Generating Pseudo-Labels from Teacher Ensemble...")

    # Accumulate predictions
    ensemble_probs = None

    for model in teacher_models:
        # Generate predictions with TTA
        preds_dict = generate_predictions(model, test_loader, device, use_tta=True)

        # Sort by rec_id to ensure alignment
        sorted_ids = sorted(preds_dict.keys())
        probs = np.array([preds_dict[rid] for rid in sorted_ids])

        if ensemble_probs is None:
            ensemble_probs = probs
        else:
            ensemble_probs += probs

    # Average
    ensemble_probs /= config.NUM_TEACHERS

    # Create DataFrame for Pseudo-Labels
    # Columns: rec_id, species_0, ..., species_18
    sorted_ids = sorted(preds_dict.keys())
    pseudo_df_data = {"rec_id": sorted_ids}
    for i in range(config.NUM_CLASSES):
        pseudo_df_data[f"species_{i}"] = ensemble_probs[:, i]

    pseudo_df = pd.DataFrame(pseudo_df_data)
    pseudo_label_path = os.path.join(config.WORKING_DIR, "pseudo_labels.parquet")
    pseudo_df.to_parquet(pseudo_label_path)
    logger.info(f"Pseudo-labels saved to {pseudo_label_path}")

    # =========================================================================
    # Stage 3: Student Training (Distillation)
    # =========================================================================
    logger.info("Starting Student Training with Combined Dataset...")

    # Reload DataLoaders with combined data
    student_train_loader, val_loader, test_loader = get_dataloaders(
        config, pseudo_labels_path=pseudo_label_path, use_combined_train=True
    )

    # Train Student
    student_model, student_path = train_model_stage(
        config,
        "student",
        student_train_loader,
        val_loader,
        device,
        logger,
        use_mixup=True,
    )

    # =========================================================================
    # Validation & Failure Analysis
    # =========================================================================
    logger.info("Performing Final Validation...")

    # Get predictions on validation set
    # We need raw targets and preds for analysis
    student_model.eval()
    all_targets = []
    all_preds = []
    all_rec_ids = []

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # Store per-sample errors
    sample_errors = []

    with torch.no_grad():
        for images, targets, rec_ids in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = student_model(images, mixup=False)
            probs = torch.sigmoid(logits)

            # Calculate loss per sample (mean over classes)
            loss = criterion(logits, targets)
            loss_per_sample = loss.mean(dim=1).cpu().numpy()
            sample_errors.extend(loss_per_sample)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())
            all_rec_ids.extend(rec_ids.numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    sample_errors = np.array(sample_errors)

    # Compute Metric
    final_metric = compute_roc_auc(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error Magnitude and Label Cardinality
    # (Hypothesis: More birds = harder to predict)
    label_cardinality = all_targets.sum(axis=1)

    if len(sample_errors) > 1:
        correlation = np.corrcoef(sample_errors, label_cardinality)[0, 1]
        print(
            f"Correlation between Error Magnitude and Label Cardinality: {correlation:.4f}"
        )
    else:
        print("Not enough samples for correlation analysis.")

    # =========================================================================
    # Submission
    # =========================================================================
    THRESHOLD = 0.9594082190886809

    if final_metric > THRESHOLD:
        logger.info("Metric threshold met. Generating submission...")

        # Generate Test Predictions
        test_preds_dict = generate_predictions(
            student_model, test_loader, device, use_tta=True
        )

        # Format Submission
        # Format: Id,Probability
        # Id = rec_id * 100 + species_id
        submission_rows = []

        sorted_test_ids = sorted(test_preds_dict.keys())
        for rid in sorted_test_ids:
            probs = test_preds_dict[rid]
            for species_id, prob in enumerate(probs):
                row_id = rid * 100 + species_id
                submission_rows.append({"Id": row_id, "Probability": prob})

        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
