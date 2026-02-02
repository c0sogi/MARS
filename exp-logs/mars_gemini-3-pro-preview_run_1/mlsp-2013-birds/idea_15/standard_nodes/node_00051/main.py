import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr

# Import from the provided library files
from library.utils import set_seed, load_checkpoint
from library.model import get_seresnet_model
from library.dataset import create_dataloaders
from library.engine import validate
from library.pipeline import (
    train_teachers,
    generate_pseudo_labels,
    train_student,
    generate_submission,
)

# --- Configuration ---
SEED = 42
BATCH_SIZE = 32
EPOCHS = 50  # Sufficient for convergence on small dataset with SWA
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUBMISSION_THRESHOLD = 0.9594082190886809
SUBMISSION_PATH = "./submission/submission.csv"


def main():
    # Ensure reproducibility
    set_seed(SEED)

    print("==================================================")
    print("   Robust Iterative Attentive SWA-Distillation    ")
    print("==================================================")

    # ---------------------------------------------------------
    # Stage 1: Train Teacher Ensemble
    # ---------------------------------------------------------
    print("\n[Stage 1] Training Teacher Ensemble (3 Models)...")
    teacher_paths = train_teachers(
        num_teachers=3, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=SEED
    )
    print(f"Teachers trained: {teacher_paths}")

    # ---------------------------------------------------------
    # Stage 2: Generation 1 Distillation (Ensemble -> Student 1)
    # ---------------------------------------------------------
    print("\n[Stage 2] Generation 1 Distillation...")

    # Generate Pseudo-labels using Teachers
    pseudo_labels_v1 = generate_pseudo_labels(
        model_paths=teacher_paths,
        output_filename="pseudo_labels_v1.parquet",
        batch_size=BATCH_SIZE,
        seed=SEED,
        load_cached_data=True,
    )

    # Train Student 1
    student_1_path = train_student(
        pseudo_labels_df=pseudo_labels_v1,
        student_name="student_1",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        seed=SEED,
    )
    print(f"Student 1 trained: {student_1_path}")

    # ---------------------------------------------------------
    # Stage 3: Generation 2 Distillation (Student 1 -> Student 2)
    # ---------------------------------------------------------
    print("\n[Stage 3] Generation 2 Distillation (Refinement)...")

    # Refine Pseudo-labels using Student 1
    pseudo_labels_v2 = generate_pseudo_labels(
        model_paths=[student_1_path],
        output_filename="pseudo_labels_v2.parquet",
        batch_size=BATCH_SIZE,
        seed=SEED,
        load_cached_data=False,  # Force regeneration with better model
    )

    # Train Student 2
    student_2_path = train_student(
        pseudo_labels_df=pseudo_labels_v2,
        student_name="student_2",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        seed=SEED,
    )
    print(f"Student 2 trained: {student_2_path}")

    # ---------------------------------------------------------
    # Evaluation & Failure Analysis
    # ---------------------------------------------------------
    print("\n[Evaluation] Validating Final Model (Student 2)...")

    # Load Validation Data
    _, val_loader, _ = create_dataloaders(batch_size=BATCH_SIZE, seed=SEED)

    # Load Final Model
    model = get_seresnet_model(num_classes=19, pretrained=False, device=DEVICE)
    load_checkpoint(model, student_2_path, device=DEVICE)
    model.eval()

    # Calculate Metric
    criterion = nn.BCEWithLogitsLoss()
    val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

    print(f"Final Validation Metric: {val_auc}")

    # --- Failure Analysis ---
    print("\n[Analysis] Performing Failure Analysis...")

    errors = []
    cardinalities = []
    energies = []

    with torch.no_grad():
        for data, target, _ in val_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            output = model(data)
            probs = torch.sigmoid(output)

            # Per-sample Mean Absolute Error
            # Shape: [Batch]
            batch_errors = torch.mean(torch.abs(probs - target), dim=1).cpu().numpy()

            # Label Cardinality (number of birds present)
            batch_cardinality = torch.sum(target, dim=1).cpu().numpy()

            # Signal Energy (mean pixel intensity)
            # data is [B, 3, H, W], normalized. We approximate energy by mean value.
            batch_energy = torch.mean(data, dim=[1, 2, 3]).cpu().numpy()

            errors.extend(batch_errors)
            cardinalities.extend(batch_cardinality)
            energies.extend(batch_energy)

    errors = np.array(errors)
    cardinalities = np.array(cardinalities)
    energies = np.array(energies)

    # Correlations
    # Handle cases where std dev is 0 (e.g. constant cardinality)
    if np.std(cardinalities) > 0:
        corr_card, _ = pearsonr(errors, cardinalities)
    else:
        corr_card = 0.0

    if np.std(energies) > 0:
        corr_energy, _ = pearsonr(errors, energies)
    else:
        corr_energy = 0.0

    print(f"Correlation (Error vs Label Cardinality): {corr_card:.4f}")
    print(f"Correlation (Error vs Signal Energy): {corr_energy:.4f}")

    if corr_card > 0.1:
        print("-> Model struggles with recordings containing multiple bird species.")
    elif corr_card < -0.1:
        print("-> Model struggles with empty/quiet recordings.")
    else:
        print("-> No strong bias found regarding number of species.")

    # ---------------------------------------------------------
    # Submission
    # ---------------------------------------------------------
    if val_auc > SUBMISSION_THRESHOLD:
        print(
            f"\nMetric ({val_auc}) > Threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        generate_submission(
            model_path=student_2_path,
            output_path=SUBMISSION_PATH,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )
    else:
        print(
            f"\nMetric ({val_auc}) <= Threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
