import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim.swa_utils import AveragedModel
from torch.optim import AdamW
import warnings

# Import library modules
from library.config import Config
from library.utils import set_seed, get_logger, calculate_roc_auc
from library.data import get_train_dataloader, get_val_dataloader, get_test_dataloader
from library.model import get_model
from library.trainer import train_one_epoch, validate, update_swa_model, finalize_swa
from library.sam import SAM

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Initialize logger
logger = get_logger("runfile")


def train_model(model_name_suffix, epochs, swa_start_epoch, use_pseudo_labels=False):
    """
    Generic training function for both Teacher and Student models.
    Handles Model setup, SAM optimizer, and SWA scheduling.
    """
    device = Config.DEVICE

    # Initialize Model (ResNet34 Pretrained)
    model = get_model(pretrained=Config.PRETRAINED).to(device)

    # Initialize SWA Model wrapper
    swa_model = AveragedModel(model).to(device)

    # Initialize Optimizer: SAM wrapping AdamW
    # SAM requires a base optimizer class and its kwargs
    base_optimizer = AdamW
    optimizer = SAM(
        model.parameters(),
        base_optimizer,
        rho=Config.SAM_RHO,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Loss Function: Multi-label classification
    criterion = nn.BCEWithLogitsLoss()

    # Data Loader
    train_loader = get_train_dataloader(use_pseudo_labels=use_pseudo_labels)

    logger.info(f"Starting training for {model_name_suffix}...")

    # Training Loop
    for epoch in range(1, epochs + 1):
        # Train for one epoch
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Update SWA model if we are past the start epoch
        if epoch >= swa_start_epoch:
            update_swa_model(swa_model, model)

    # Finalize SWA: Update BatchNorm statistics using training data
    finalize_swa(swa_model, train_loader, device)

    return swa_model


def predict_with_tta(model, dataloader):
    """
    Performs inference with Test-Time Augmentation (Horizontal Flip).
    Returns raw probabilities (sigmoid applied) and recording IDs.
    """
    device = Config.DEVICE
    model.eval()

    all_probs = []
    all_rec_ids = []

    with torch.no_grad():
        for inputs, rec_ids in dataloader:
            inputs = inputs.to(device)

            # Forward pass 1: Original Image
            outputs_1 = model(inputs)
            probs_1 = torch.sigmoid(outputs_1)

            # Forward pass 2: Horizontal Flip (Time Inversion)
            # Input shape is (B, C, H, W). We flip the last dimension (Width/Time).
            inputs_flipped = torch.flip(inputs, dims=[-1])
            outputs_2 = model(inputs_flipped)
            probs_2 = torch.sigmoid(outputs_2)

            # Average the probabilities
            avg_probs = (probs_1 + probs_2) / 2.0

            all_probs.append(avg_probs.cpu().numpy())
            all_rec_ids.extend(rec_ids.numpy())

    return np.vstack(all_probs), np.array(all_rec_ids)


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)
    Config.setup()
    # Config.print_summary() # Optional, keeping output clean

    # =========================================================================
    # STAGE 1: TEACHER ENSEMBLE TRAINING
    # =========================================================================
    logger.info(">>> STAGE 1: Training Teacher Ensemble")

    teachers = []
    for i in range(Config.NUM_TEACHERS):
        logger.info(f"Training Teacher {i+1}/{Config.NUM_TEACHERS}")

        # Train independent teacher model
        teacher_model = train_model(
            model_name_suffix=f"Teacher_{i}",
            epochs=Config.EPOCHS_TEACHER,
            swa_start_epoch=Config.SWA_START_EPOCH_TEACHER,
            use_pseudo_labels=False,
        )
        teachers.append(teacher_model)

    # =========================================================================
    # STAGE 2: PSEUDO-LABEL GENERATION
    # =========================================================================
    logger.info(">>> STAGE 2: Generating Pseudo-Labels")

    test_loader = get_test_dataloader()

    ensemble_probs = []
    rec_ids = None

    # Generate predictions from each teacher
    for i, teacher in enumerate(teachers):
        probs, ids = predict_with_tta(teacher, test_loader)
        ensemble_probs.append(probs)
        if rec_ids is None:
            rec_ids = ids

    # Average predictions across the ensemble
    avg_probs = np.mean(ensemble_probs, axis=0)

    # Create DataFrame for pseudo-labels
    # Columns must be strings "0", "1", ... matching class indices
    df_pseudo = pd.DataFrame(
        avg_probs, columns=[str(i) for i in range(Config.NUM_CLASSES)]
    )
    df_pseudo["rec_id"] = rec_ids

    # Save to Parquet for the Student to load
    df_pseudo.to_parquet(Config.PSEUDO_LABEL_PATH)
    logger.info(f"Pseudo-labels saved to {Config.PSEUDO_LABEL_PATH}")

    # Clean up memory
    del teachers
    torch.cuda.empty_cache()

    # =========================================================================
    # STAGE 3: STUDENT TRAINING
    # =========================================================================
    logger.info(">>> STAGE 3: Training Student Model")

    # Train Student on Combined (Labeled + Pseudo-Labeled) Data
    student_model = train_model(
        model_name_suffix="Student",
        epochs=Config.EPOCHS_STUDENT,
        swa_start_epoch=Config.SWA_START_EPOCH_STUDENT,
        use_pseudo_labels=True,
    )

    # =========================================================================
    # VALIDATION & FAILURE ANALYSIS
    # =========================================================================
    logger.info(">>> VALIDATION & FAILURE ANALYSIS")

    val_loader = get_val_dataloader()
    criterion = nn.BCEWithLogitsLoss()

    # Evaluate Student Model on Validation Set
    val_loss, val_auc = validate(student_model, val_loader, criterion, Config.DEVICE)

    # Print required metric
    print(f"Final Validation Metric: {val_auc}")

    # --- Failure Analysis ---
    # Calculate correlation between Error Magnitude and Spectrogram Intensity
    student_model.eval()
    all_preds = []
    all_targets = []
    all_inputs_mean = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(Config.DEVICE)

            # Forward pass
            outputs = student_model(inputs)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

            # Feature extraction: Mean pixel intensity of the batch
            # inputs shape: (B, 3, H, W) -> mean over dims 1,2,3
            means = inputs.mean(dim=[1, 2, 3]).cpu().numpy()
            all_inputs_mean.extend(means)

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_inputs_mean = np.array(all_inputs_mean)

    # Error Magnitude: Mean Absolute Error per sample (averaged over classes)
    error_magnitude = np.abs(all_preds - all_targets).mean(axis=1)

    # Calculate Correlation
    if len(error_magnitude) > 1:
        correlation = np.corrcoef(error_magnitude, all_inputs_mean)[0, 1]
        print(
            f"Correlation between Error Magnitude and Spectrogram Intensity: {correlation}"
        )
    else:
        print("Not enough samples for correlation analysis.")

    # =========================================================================
    # SUBMISSION
    # =========================================================================
    THRESHOLD = 0.9594082190886809

    if val_auc > THRESHOLD:
        logger.info(">>> GENERATING SUBMISSION")

        # Predict on Test Set with Student (using TTA)
        test_probs, test_ids = predict_with_tta(student_model, test_loader)

        # Format Submission
        # Format: Id,Probability where Id = rec_id * 100 + species_id
        submission_rows = []
        for i in range(len(test_ids)):
            rec_id = test_ids[i]
            probs = test_probs[i]

            for species_idx, prob in enumerate(probs):
                row_id = int(rec_id * 100 + species_idx)
                submission_rows.append({"Id": row_id, "Probability": prob})

        df_sub = pd.DataFrame(submission_rows)

        # Save Submission
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        logger.info(
            f"Validation metric {val_auc} did not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
