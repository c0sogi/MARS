import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import cv2
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.models import create_model
from library.engine import fit, validate, inference
from library.distillation import generate_pseudo_labels


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Train Heterogeneous Teacher Ensemble
    # Teachers: 2x ResNet-34, 2x DenseNet-121
    # We train them on the labeled training set (Fold 0)

    print("\n=== Stage 1: Training Heterogeneous Teacher Ensemble ===")

    # Get standard dataloaders (only labeled train data)
    dataloaders = get_dataloaders(pseudo_labels_path=None)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    for i, arch in enumerate(Config.TEACHER_ARCHS):
        print(f"\nTraining Teacher {i+1}/{len(Config.TEACHER_ARCHS)}: {arch}")

        # Initialize Model
        model = create_model(arch, num_classes=Config.NUM_CLASSES, pretrained=True)
        model.to(device)

        # Optimizer & Scheduler
        # Using Adam with Cosine Annealing. eta_min set to SWA_LR to stabilize for SWA.
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS, eta_min=Config.SWA_LR
        )

        # Define save path
        save_path = Config.get_teacher_path(i, arch)

        # Train
        fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=Config.EPOCHS,
            swa_start_epoch=Config.SWA_START_EPOCH_TEACHER,
            save_path=save_path,
            patience=None,  # Disable early stopping to ensure full SWA schedule
        )

    # 3. Generate Pseudo-Labels
    print("\n=== Stage 2: Generating Pseudo-Labels ===")
    # Force regeneration to use the models we just trained (load_cached_data=False)
    # The function saves the parquet file to Config.PSEUDO_LABELS_PATH
    generate_pseudo_labels(load_cached_data=False)

    # 4. Train Student Model
    print("\n=== Stage 3: Training Student Model ===")

    # Reload dataloaders with pseudo-labels injected
    # This merges Train (Hard Labels) + Test (Soft Labels)
    student_dataloaders = get_dataloaders(pseudo_labels_path=Config.PSEUDO_LABELS_PATH)
    student_train_loader = student_dataloaders["train"]
    student_val_loader = student_dataloaders["val"]  # Same validation set

    # Initialize Student (ResNet-34)
    student_model = create_model(
        Config.STUDENT_ARCH, num_classes=Config.NUM_CLASSES, pretrained=True
    )
    student_model.to(device)

    # Optimizer & Scheduler for Student
    optimizer = optim.Adam(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.SWA_LR
    )

    student_save_path = Config.get_student_path()

    # Train Student
    final_student_model = fit(
        model=student_model,
        train_loader=student_train_loader,
        val_loader=student_val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        swa_start_epoch=Config.SWA_START_EPOCH_STUDENT,
        save_path=student_save_path,
        patience=None,
    )

    # 5. Final Validation & Metrics
    print("\n=== Stage 4: Final Evaluation ===")

    # Ensure we are using the best/final SWA model
    final_student_model.eval()
    val_loss, val_auc, val_preds = validate(
        final_student_model, student_val_loader, device
    )

    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\n=== Stage 5: Failure Analysis ===")

    # Calculate per-sample error
    # We need the targets from the validation set
    val_targets = []
    val_rec_ids = []

    # Iterate loader to get targets and IDs aligned with predictions
    # Note: validate() returns preds concatenated, but we need to match them to metadata
    # The val_loader is sequential (shuffle=False), so order is preserved.

    for batch in student_val_loader:
        val_targets.append(batch["label"].numpy())
        val_rec_ids.append(batch["rec_id"].numpy())

    val_targets = np.concatenate(val_targets)
    val_rec_ids = np.concatenate(val_rec_ids)

    # Compute Mean Absolute Error per sample
    # Shape: (N, 19)
    abs_errors = np.abs(val_preds - val_targets)
    mean_sample_error = np.mean(
        abs_errors, axis=1
    )  # Average error across classes for each sample

    # Load image metadata to correlate features
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # We need to map rec_id to file path to load image and compute intensity
    # val_df has 'rec_id' and 'file_path'

    intensities = []
    aligned_errors = []

    print("Computing correlation between Error and Signal Intensity...")

    for i, rid in enumerate(val_rec_ids):
        # Find path
        row = val_df[val_df["rec_id"] == rid]
        if row.empty:
            continue

        rel_path = row.iloc[0]["file_path"]
        wav_basename = os.path.basename(rel_path)
        bmp_basename = os.path.splitext(wav_basename)[0] + ".bmp"
        img_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_basename)

        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                intensity = np.mean(img) / 255.0
                intensities.append(intensity)
                aligned_errors.append(mean_sample_error[i])

    if len(intensities) > 1:
        corr, _ = pearsonr(aligned_errors, intensities)
        print(f"Correlation (Error vs. Image Intensity): {corr:.4f}")
        if corr < 0:
            print("Observation: Higher signal intensity correlates with lower error.")
        else:
            print("Observation: Higher signal intensity correlates with higher error.")
    else:
        print("Insufficient data for correlation analysis.")

    # 7. Submission
    # Threshold: 0.9594082190886809
    threshold = 0.9594082190886809

    if val_auc > threshold:
        print(
            f"\nValidation AUC ({val_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Load test loader
        test_loader = dataloaders["test"]

        inference(
            model=final_student_model,
            test_loader=test_loader,
            device=device,
            submission_path=Config.SUBMISSION_PATH,
        )
    else:
        print(
            f"\nValidation AUC ({val_auc}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
