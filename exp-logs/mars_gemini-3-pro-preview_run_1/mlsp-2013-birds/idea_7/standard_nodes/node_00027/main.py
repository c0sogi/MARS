import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, save_model, load_model, compute_auc
from library.dataset import load_metadata, get_dataloaders
from library.model import BirdResNet34
from library.training import Trainer, validate
from library.distillation import generate_ensemble_pseudo_labels

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 2. Load Metadata
    train_df, val_df, test_df = load_metadata()

    # 3. Stage 1: Train Teacher Ensemble
    # We train multiple teachers to create stable pseudo-labels
    teachers = []

    # Get dataloaders for training teachers (only on labeled train_df)
    train_loader_teacher, val_loader, test_loader_unlabeled = get_dataloaders(
        train_df=train_df, val_df=val_df, test_df=test_df, batch_size=Config.BATCH_SIZE
    )

    print(f"--- Stage 1: Training {Config.NUM_TEACHERS} Teacher Models ---")

    for i in range(Config.NUM_TEACHERS):
        seed = Config.TEACHER_SEEDS[i]
        set_seed(seed)  # Ensure diversity via initialization and data shuffling

        print(f"Training Teacher {i+1}/{Config.NUM_TEACHERS} (Seed {seed})")

        model = BirdResNet34(pretrained=True).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        # Cosine Schedule for smooth convergence
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
        criterion = nn.BCEWithLogitsLoss()

        trainer = Trainer(
            model=model,
            train_loader=train_loader_teacher,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
        )

        # Train and retrieve best model
        best_teacher, _ = trainer.fit(num_epochs=Config.EPOCHS)

        # Save to working directory
        teacher_path = os.path.join(Config.WORKING_DIR, f"teacher_{i}.pth")
        save_model(best_teacher, teacher_path)
        teachers.append(best_teacher)

    # 4. Stage 2: Distillation (Pseudo-labeling)
    print("--- Stage 2: Generating Pseudo-Labels ---")
    # Generate soft labels for the test set using the teacher ensemble
    pseudo_labels_df = generate_ensemble_pseudo_labels(
        teachers=teachers,
        test_loader=test_loader_unlabeled,
        device=device,
        load_cached_data=True,
    )

    # 5. Stage 3: Train Student Model
    print("--- Stage 3: Training Student Model ---")

    # Prepare Combined Data (Train Hard Labels + Test Soft Labels)

    # Define label columns
    label_cols = [f"species_{k}" for k in range(Config.NUM_CLASSES)]

    # Merge pseudo labels with test_df to ensure alignment by rec_id
    # Drop dummy species columns from test_df to avoid column name collision (e.g., species_0_x, species_0_y)
    test_df_clean = test_df.drop(columns=label_cols, errors="ignore")
    test_df_with_labels = pd.merge(
        test_df_clean, pseudo_labels_df, on="rec_id", how="left"
    )

    # Extract targets
    train_targets = train_df[label_cols].values.astype(np.float32)
    test_targets = test_df_with_labels[label_cols].values.astype(np.float32)

    # Combine DataFrames and Targets
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_targets = np.vstack([train_targets, test_targets])

    # Create Student Dataloader
    # We pass the combined dataframe and the combined soft label array
    student_train_loader, _, _ = get_dataloaders(
        train_df=combined_df,
        val_df=val_df,  # Use original validation set
        batch_size=Config.BATCH_SIZE,
        train_soft_labels=combined_targets,
    )

    # Initialize Student Model
    set_seed(Config.SEED)  # Reset seed for student training
    student_model = BirdResNet34(pretrained=True).to(device)
    optimizer = optim.AdamW(student_model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(
        model=student_model,
        train_loader=student_train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
    )

    # Train Student
    best_student, history = trainer.fit(num_epochs=Config.EPOCHS)

    # Save Student
    student_path = os.path.join(Config.WORKING_DIR, "best_student.pth")
    save_model(best_student, student_path)

    # 6. Evaluation & Failure Analysis
    print("--- Evaluation & Failure Analysis ---")

    # Compute Final Validation Metric
    val_loss, val_auc = validate(best_student, val_loader, criterion, device)
    print(f"Final Validation Metric: {val_auc}")

    # Failure Analysis
    # We correlate prediction error with input signal energy (mean pixel intensity)
    best_student.eval()
    errors = []
    energies = []

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = best_student(images)
            probs = torch.sigmoid(logits)

            # Calculate Mean Absolute Error per sample (averaged across species)
            batch_errors = torch.abs(probs - targets).mean(dim=1).cpu().numpy()
            errors.extend(batch_errors)

            # Calculate Signal Energy: Mean intensity of the input image
            # Images are (Batch, 3, H, W). We average over dimensions (1, 2, 3)
            batch_energies = images.mean(dim=(1, 2, 3)).cpu().numpy()
            energies.extend(batch_energies)

    errors = np.array(errors)
    energies = np.array(energies)

    if len(errors) > 1:
        # Compute Pearson correlation
        correlation = np.corrcoef(errors, energies)[0, 1]
        print(
            f"Correlation between Error Magnitude and Input Signal Energy: {correlation:.16f}"
        )
    else:
        print("Insufficient samples for failure analysis.")

    # 7. Submission
    THRESHOLD = 0.9255537489325414

    if val_auc > THRESHOLD:
        print("Validation metric meets threshold. Generating submission...")

        # Predict on Test Set (using the unlabeled loader which iterates test_df)
        best_student.eval()
        test_preds = []
        test_ids = []

        with torch.no_grad():
            for images, _, rec_ids in test_loader_unlabeled:
                images = images.to(device)
                logits = best_student(images)
                probs = torch.sigmoid(logits)

                test_preds.append(probs.cpu().numpy())
                test_ids.extend(rec_ids.numpy())

        if len(test_preds) > 0:
            test_preds = np.vstack(test_preds)

        # Format Submission: Id,Probability
        # Id = rec_id * 100 + species_id
        submission_rows = []
        for i, rec_id in enumerate(test_ids):
            rec_id = int(rec_id)
            probs = test_preds[i]
            for species_id, prob in enumerate(probs):
                row_id = rec_id * 100 + species_id
                submission_rows.append({"Id": row_id, "Probability": prob})

        submission_df = pd.DataFrame(submission_rows)

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {val_auc} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
