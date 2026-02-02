import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

# Import provided library functions
from library.utils import set_seed
from library.data import get_dataloaders, BirdDataset, get_transforms
from library.model import create_bird_model
from library.training import run_training_schedule, validate
from library.inference import predict_ensemble, run_inference, generate_submission


def main():
    # --- Configuration ---
    # Fast baseline configuration
    BATCH_SIZE = 32
    # Cite solution_lesson_node_00029: Scaling Training Horizons for Regularized Convergence
    EPOCHS = 60
    LR = 1e-3
    SWA_START_PCT = 0.70
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SUBMISSION_THRESHOLD = 0.9594082190886809

    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_25")

    os.makedirs(IDEA_DIR, exist_ok=True)

    # Set global seed for reproducibility
    set_seed(42)

    print(f"Using device: {DEVICE}")

    # --- Stage 1: Train Teacher Ensemble ---
    print("\n=== Stage 1: Training Teacher Ensemble ===")

    # Get standard dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_dir=METADATA_DIR,
        input_dir=INPUT_DIR,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )

    teacher_models = []
    num_teachers = 3

    for i in range(num_teachers):
        print(f"\nTraining Teacher {i+1}/{num_teachers}...")

        # Initialize model
        # We use a different seed for each teacher initialization if not pretrained,
        # but here we use pretrained=True, so diversity comes from data shuffling and mixup.
        # To ensure some diversity, we can re-seed before creation or just rely on loader randomness.
        set_seed(42 + i)
        model = create_bird_model(num_classes=19, pretrained=True)

        checkpoint_dir = os.path.join(IDEA_DIR, f"teacher_{i}")

        # Train
        _, swa_model = run_training_schedule(
            model,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            swa_start_epoch_pct=SWA_START_PCT,
            lr=LR,
            device=DEVICE,
            checkpoint_dir=checkpoint_dir,
        )

        teacher_models.append(swa_model)

    # --- Stage 2: Generate Pseudo-Labels ---
    print("\n=== Stage 2: Generating Pseudo-Labels ===")

    # Generate predictions on test set using Teacher Ensemble
    # predict_ensemble handles TTA and averaging internally
    rec_ids, avg_probs = predict_ensemble(teacher_models, test_loader, device=DEVICE)

    # Create dictionary mapping rec_id to soft labels
    pseudo_labels_dict = {int(rid): prob for rid, prob in zip(rec_ids, avg_probs)}

    print(f"Generated pseudo-labels for {len(pseudo_labels_dict)} test samples.")

    # --- Stage 3: Train Student Model ---
    print("\n=== Stage 3: Training Student Model ===")

    # Create Combined Dataset (Train + Test)
    # Load metadata manually to combine
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Concatenate
    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    # Create Dataset and Loader
    # Note: We use the training transform for the student
    train_transform = get_transforms(phase="train")

    combined_dataset = BirdDataset(
        df_combined,
        INPUT_DIR,
        transform=train_transform,
        pseudo_labels=pseudo_labels_dict,  # Pass pseudo-labels here
        num_classes=19,
    )

    combined_loader = DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Initialize Student Model
    set_seed(999)  # Distinct seed for student
    student_model = create_bird_model(num_classes=19, pretrained=True)

    student_checkpoint_dir = os.path.join(IDEA_DIR, "student")

    # Train Student
    _, student_swa_model = run_training_schedule(
        student_model,
        combined_loader,  # Use combined loader
        val_loader,  # Validate on original hold-out val set
        epochs=EPOCHS,
        swa_start_epoch_pct=SWA_START_PCT,
        lr=LR,
        device=DEVICE,
        checkpoint_dir=student_checkpoint_dir,
    )

    # --- Evaluation & Failure Analysis ---
    print("\n=== Evaluation & Failure Analysis ===")

    # 1. Final Validation Metric
    criterion = nn.BCEWithLogitsLoss()
    val_metrics = validate(student_swa_model, val_loader, criterion, DEVICE)
    final_auc = val_metrics["auc"]

    # REQUIRED FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 2. Failure Analysis
    # We will compute the correlation between error (Log Loss) and Label Cardinality (Number of Species)
    student_swa_model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = student_swa_model(inputs)
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    # Compute error per sample (mean log loss across classes)
    # Clip preds for stability
    eps = 1e-15
    preds_clipped = np.clip(all_preds, eps, 1 - eps)

    # Calculate binary cross entropy for each sample (averaged over classes)
    # shape: (N_samples, N_classes) -> (N_samples,)
    sample_losses = -np.mean(
        all_targets * np.log(preds_clipped)
        + (1 - all_targets) * np.log(1 - preds_clipped),
        axis=1,
    )

    # Calculate label cardinality (number of active species)
    label_counts = np.sum(all_targets, axis=1)

    # Correlation
    if len(sample_losses) > 1:
        correlation = np.corrcoef(sample_losses, label_counts)[0, 1]
    else:
        correlation = 0.0

    print(
        f"Failure Analysis - Correlation between Error and Label Cardinality: {correlation:.4f}"
    )
    if correlation > 0:
        print(
            "-> Positive correlation: The model struggles more with samples containing multiple bird species."
        )
    else:
        print(
            "-> Negative/Zero correlation: Error is not strongly driven by the number of species present."
        )

    # --- Submission ---
    if final_auc > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        # Run inference on test set using the Student SWA model
        # We use the provided run_inference wrapper which handles TTA and formatting
        run_inference(
            student_swa_model,
            test_loader,
            device=DEVICE,
            output_dir="./submission",
            filename="submission.csv",
        )
    else:
        print(
            f"\nValidation metric ({final_auc}) did not exceed threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
