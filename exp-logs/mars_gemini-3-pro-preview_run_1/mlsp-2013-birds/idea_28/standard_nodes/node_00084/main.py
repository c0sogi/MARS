import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import set_seed, compute_auc
from library.data import get_dataloaders
from library.training import run_training_cycle, validate
from library.inference import generate_pseudo_labels, generate_submission


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)

    # Configuration for Fast Baseline Execution
    # We reduce epochs to 30 to ensure the pipeline fits comfortably within the time limit
    # while still allowing sufficient convergence for the small dataset (208 samples).
    FAST_EPOCHS = 30
    TEACHER_SWA_START = int(FAST_EPOCHS * 0.75)  # Start SWA at 75%
    STUDENT_SWA_START = int(FAST_EPOCHS * 0.70)  # Start SWA at 70%

    # Load Metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        print(f"Error: Metadata file not found at {Config.TRAIN_METADATA_PATH}")
        return

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # ==========================================
    # Stage 1: Train Augmentation-Stratified Teacher Ensemble
    # ==========================================
    print("Starting Stage 1: Teacher Ensemble Training...")

    # Define the policies for the ensemble members
    policies = ["Texture", "Feature", "Balanced"]
    teacher_checkpoints = []

    for i, policy in enumerate(policies):
        teacher_name = f"teacher_{i}_{policy}"

        # Get Dataloaders configured for the specific policy
        loaders = get_dataloaders(train_df, val_df, test_df, teacher_policy=policy)

        # Train the teacher model
        # run_training_cycle handles the full loop, SWA, and saving checkpoints
        _ = run_training_cycle(
            model_name=teacher_name,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            mixup_alpha=Config.STRATIFIED_POLICIES[policy]["mixup_alpha"],
            swa_start_epoch=TEACHER_SWA_START,
            num_epochs=FAST_EPOCHS,
            device=Config.DEVICE,
        )

        # Store the filename of the SWA checkpoint for the distillation stage
        teacher_checkpoints.append(f"{teacher_name}_swa.pth")

    # ==========================================
    # Stage 2: Generate Calibrated Pseudo-Labels
    # ==========================================
    print("Starting Stage 2: Pseudo-Label Generation...")

    # Get a loader for the test set (transforms are standard validation transforms)
    loaders = get_dataloaders(train_df, val_df, test_df, teacher_policy="Balanced")

    # Generate pseudo-labels using the ensemble
    # This function handles loading models, TTA, Temperature Scaling, and Averaging
    pseudo_labels_df = generate_pseudo_labels(
        teacher_checkpoints,
        loaders["test"],
        test_df,
        Config.DEVICE,
        load_cached_data=False,  # Force regeneration to use the newly trained models
    )

    # ==========================================
    # Stage 3: Train Student Model
    # ==========================================
    print("Starting Stage 3: Student Training...")
    student_name = "student_resnet34"

    # Get Dataloaders with Pseudo-Labels
    # This automatically concatenates the Labeled Train set and the Pseudo-Labeled Test set
    student_loaders = get_dataloaders(
        train_df,
        val_df,
        test_df,
        pseudo_labels_df=pseudo_labels_df,
        student_policy=Config.STUDENT_POLICY_NAME,
    )

    # Train the Student model
    student_swa_model = run_training_cycle(
        model_name=student_name,
        train_loader=student_loaders["train"],
        val_loader=student_loaders["val"],
        mixup_alpha=Config.STRATIFIED_POLICIES[Config.STUDENT_POLICY_NAME][
            "mixup_alpha"
        ],
        swa_start_epoch=STUDENT_SWA_START,
        num_epochs=FAST_EPOCHS,
        device=Config.DEVICE,
    )

    # ==========================================
    # Stage 4: Validation & Failure Analysis
    # ==========================================
    print("Starting Stage 4: Validation & Failure Analysis...")

    # 1. Compute Final Validation Metric
    criterion = torch.nn.BCEWithLogitsLoss()
    val_loss, val_auc = validate(
        student_swa_model, student_loaders["val"], criterion, Config.DEVICE
    )

    # Print metric with full precision
    print(f"Final Validation Metric: {val_auc}")

    # 2. Failure Analysis
    # We correlate the model's error magnitude with the Label Cardinality (number of species)
    student_swa_model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in student_loaders["val"]:
            images = images.to(Config.DEVICE)
            outputs = student_swa_model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Mean Absolute Error per sample across all classes
    errors = np.abs(all_targets - all_preds).mean(axis=1)

    # Calculate Label Cardinality (Input Feature)
    cardinality = all_targets.sum(axis=1)

    # Compute Pearson Correlation
    if np.std(errors) > 1e-9 and np.std(cardinality) > 1e-9:
        corr = np.corrcoef(errors, cardinality)[0, 1]
    else:
        corr = 0.0

    print(f"Correlation between Error and Label Cardinality: {corr}")

    # ==========================================
    # Stage 5: Submission
    # ==========================================
    THRESHOLD = 0.9594082190886809

    if val_auc > THRESHOLD:
        print("Validation metric meets threshold. Generating submission...")
        generate_submission(
            student_swa_model, student_loaders["test"], test_df, Config.DEVICE
        )
    else:
        print(f"Validation metric {val_auc} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
