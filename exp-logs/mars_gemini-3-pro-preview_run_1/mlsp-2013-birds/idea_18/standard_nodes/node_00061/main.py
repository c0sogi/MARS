import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.utils import set_seed, calculate_roc_auc
from library.model import create_model
from library.dataset import get_dataloaders
from library.core import Trainer, generate_submission
from library.distillation import generate_pseudo_labels


def main():
    # --- Configuration ---
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Teacher Config
    NUM_TEACHERS = 3
    TEACHER_EPOCHS = 50
    TEACHER_SWA_START = 38  # SWA for last ~25%
    TEACHER_LR = 1e-3

    # Student Config
    STUDENT_EPOCHS = 50
    STUDENT_SWA_START = 35  # SWA for last ~30%
    STUDENT_LR = 1e-3
    STUDENT_DROP_PATH = 0.1
    STUDENT_HEAD_DROPOUT = 0.5

    # Paths
    WORKING_DIR = "./working/idea_18"
    PSEUDO_LABEL_PATH = os.path.join(WORKING_DIR, "pseudo_labels.parquet")
    SUBMISSION_PATH = "./submission/submission.csv"
    THRESHOLD = 0.9594082190886809

    set_seed(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)

    print(f"Running on device: {DEVICE}")

    # =========================================================================
    # STAGE 1: Train Teacher Ensemble
    # =========================================================================
    print("\n=== STAGE 1: Training Teacher Ensemble ===")

    teacher_models = []

    # Get standard dataloaders (only labeled data)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        load_cached_data=True,
        use_mixup=False,  # Teachers train on clean data
    )

    for i in range(NUM_TEACHERS):
        print(f"\nTraining Teacher {i+1}/{NUM_TEACHERS}...")

        # Vanilla ResNet-34 (No stochastic depth, no head dropout)
        model = create_model(
            num_classes=19, pretrained=True, drop_path_rate=0.0, head_dropout=0.0
        ).to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=TEACHER_LR)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TEACHER_EPOCHS
        )

        checkpoint_dir = os.path.join(WORKING_DIR, "checkpoints", f"teacher_{i}")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            scheduler=scheduler,
            checkpoint_dir=checkpoint_dir,
            use_swa=True,
            swa_start_epoch=TEACHER_SWA_START,
        )

        trainer.fit(train_loader, val_loader, epochs=TEACHER_EPOCHS, patience=15)

        # Load best SWA model for distillation
        # If SWA model exists, use it; otherwise use best model
        swa_path = os.path.join(checkpoint_dir, "model_swa.pth")
        if os.path.exists(swa_path):
            model.load_state_dict(torch.load(swa_path, map_location=DEVICE))
        else:
            # Fallback to best model
            best_path = os.path.join(checkpoint_dir, "model_best.pth")
            if os.path.exists(best_path):
                checkpoint = torch.load(best_path, map_location=DEVICE)
                model.load_state_dict(checkpoint["state_dict"])

        teacher_models.append(model)

    # =========================================================================
    # STAGE 2: Distillation (Pseudo-Label Generation)
    # =========================================================================
    print("\n=== STAGE 2: Generating Pseudo-Labels ===")

    generate_pseudo_labels(
        teacher_models=teacher_models,
        test_loader=test_loader,
        device=DEVICE,
        output_path=PSEUDO_LABEL_PATH,
        load_cached_data=False,  # Force regeneration
    )

    # Free up memory
    del teacher_models
    del train_loader
    torch.cuda.empty_cache()

    # =========================================================================
    # STAGE 3: Train Noisy Student
    # =========================================================================
    print("\n=== STAGE 3: Training Noisy Student ===")

    # Get dataloaders with pseudo-labels mixed in
    # This effectively expands the training set
    train_loader_student, val_loader_student, test_loader_student = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pseudo_label_path=PSEUDO_LABEL_PATH,
        load_cached_data=False,  # Reload to include pseudo labels
        use_mixup=True,  # Use Mixup for student regularization
        mixup_alpha=0.2,
    )

    # Noisy Student Model (Stochastic Depth + Head Dropout)
    student_model = create_model(
        num_classes=19,
        pretrained=True,
        drop_path_rate=STUDENT_DROP_PATH,
        head_dropout=STUDENT_HEAD_DROPOUT,
    ).to(DEVICE)

    optimizer = optim.Adam(student_model.parameters(), lr=STUDENT_LR)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STUDENT_EPOCHS)

    student_checkpoint_dir = os.path.join(WORKING_DIR, "checkpoints", "student")

    student_trainer = Trainer(
        model=student_model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        scheduler=scheduler,
        checkpoint_dir=student_checkpoint_dir,
        use_swa=True,
        swa_start_epoch=STUDENT_SWA_START,
    )

    student_trainer.fit(
        train_loader_student, val_loader_student, epochs=STUDENT_EPOCHS, patience=15
    )

    # =========================================================================
    # STAGE 4: Validation & Failure Analysis
    # =========================================================================
    print("\n=== STAGE 4: Validation & Failure Analysis ===")

    # Load best SWA Student model
    swa_path = os.path.join(student_checkpoint_dir, "model_swa.pth")
    if os.path.exists(swa_path):
        print("Loading SWA Student model for validation...")
        student_model.load_state_dict(torch.load(swa_path, map_location=DEVICE))
    else:
        print("SWA model not found, loading best model...")
        best_path = os.path.join(student_checkpoint_dir, "model_best.pth")
        checkpoint = torch.load(best_path, map_location=DEVICE)
        student_model.load_state_dict(checkpoint["state_dict"])

    student_model.eval()

    # 1. Compute Metric
    _, val_auc = student_trainer.validate(val_loader_student, student_model)
    print(f"Final Validation Metric: {val_auc}")

    # 2. Failure Analysis
    print("Performing Failure Analysis...")
    all_errors = []
    all_img_means = []
    all_img_stds = []

    # We iterate manually to get images and targets
    with torch.no_grad():
        for images, labels, _ in val_loader_student:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = student_model(images)
            probs = torch.sigmoid(outputs)

            # Calculate mean absolute error per sample (averaged across classes)
            # Shape: (B,)
            error = torch.abs(labels - probs).mean(dim=1).cpu().numpy()
            all_errors.extend(error)

            # Calculate image stats
            # images shape: (B, 3, H, W)
            # Mean intensity per image
            img_mean = images.mean(dim=(1, 2, 3)).cpu().numpy()
            all_img_means.extend(img_mean)

            # Std intensity per image
            img_std = images.std(dim=(1, 2, 3)).cpu().numpy()
            all_img_stds.extend(img_std)

    all_errors = np.array(all_errors)
    all_img_means = np.array(all_img_means)
    all_img_stds = np.array(all_img_stds)

    # Correlations
    corr_mean = np.corrcoef(all_errors, all_img_means)[0, 1]
    corr_std = np.corrcoef(all_errors, all_img_stds)[0, 1]

    print(f"Correlation (Error vs Image Mean Intensity): {corr_mean:.4f}")
    print(f"Correlation (Error vs Image Contrast/Std): {corr_std:.4f}")

    if abs(corr_mean) > 0.2:
        print(">> Analysis: Model performance is sensitive to signal brightness.")
    if abs(corr_std) > 0.2:
        print(">> Analysis: Model performance is sensitive to signal contrast.")

    # =========================================================================
    # STAGE 5: Submission
    # =========================================================================
    if val_auc > THRESHOLD:
        print(
            f"\nValidation Metric ({val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        # Note: test_loader_student is the same as original test_loader
        ids, predictions = student_trainer.predict(
            test_loader_student, use_swa_model=True
        )

        generate_submission(ids, predictions, output_path=SUBMISSION_PATH)
    else:
        print(
            f"\nValidation Metric ({val_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
