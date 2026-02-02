import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, calculate_roc_auc, sanitize_pseudo_labels
from library.dataset import get_dataloader, BirdDataset, MixupCollate
from library.model import BirdResNet
from library.trainer import Trainer


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading DataLoaders...")
    train_loader = get_dataloader("train", batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader("val", batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader("test", batch_size=Config.BATCH_SIZE, shuffle=False)

    # ==========================
    # Stage 1: Teacher Ensemble
    # ==========================
    teacher_preds = []
    teacher_ids = None

    for i in range(Config.NUM_TEACHERS):
        print(f"\n=== Training Teacher {i+1}/{Config.NUM_TEACHERS} ===")

        # Initialize Model, Criterion, Optimizer
        model = BirdResNet(pretrained=Config.PRETRAINED).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

        # Scheduler for standard phase (before SWA)
        # Trainer steps scheduler once per epoch
        swa_start = Config.get_swa_start_epoch(
            Config.TEACHER_EPOCHS, Config.TEACHER_SWA_START_RATIO
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=swa_start
        )

        trainer = Trainer(model, device, criterion, optimizer, scheduler)
        save_path = os.path.join(Config.WORKING_DIR, f"teacher_{i}.pth")

        # Train with SWA
        swa_model = trainer.fit_swa(
            train_loader,
            val_loader,
            total_epochs=Config.TEACHER_EPOCHS,
            swa_start_epoch=swa_start,
            swa_lr=Config.TEACHER_SWA_LR,
            save_path=save_path,
        )

        # Predict on Test Set (Fold 1) for Pseudo-Labeling
        print(f"Generating predictions with Teacher {i+1}...")
        swa_model.eval()
        preds = []
        ids = []
        with torch.no_grad():
            for images, _, rec_ids in test_loader:
                images = images.to(device)
                outputs = swa_model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds.append(probs)
                ids.append(rec_ids.numpy())

        teacher_preds.append(np.concatenate(preds))
        if teacher_ids is None:
            teacher_ids = np.concatenate(ids)

    # ==========================
    # Stage 2: Pseudo-Labeling
    # ==========================
    print("\n=== Generating Pseudo-Labels ===")
    # Average predictions across teachers
    avg_preds = np.mean(teacher_preds, axis=0)
    sanitized_preds = sanitize_pseudo_labels(avg_preds)

    # Map rec_id to probability vector
    pseudo_labels_dict = {rid: pred for rid, pred in zip(teacher_ids, sanitized_preds)}
    print(f"Generated pseudo-labels for {len(pseudo_labels_dict)} test samples.")

    # ==========================
    # Stage 3: Student Training
    # ==========================
    print("\n=== Training Student Model ===")

    # Prepare Combined Dataset (Train + Test with Pseudo-labels)
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    # Initialize Student Dataset and Loader
    student_dataset = BirdDataset(
        df_combined, mode="train", pseudo_labels=pseudo_labels_dict
    )

    # Use Mixup Collate
    collate_fn = MixupCollate(alpha=Config.MIXUP_ALPHA)

    student_loader = DataLoader(
        student_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Initialize Student Model
    model = BirdResNet(pretrained=Config.PRETRAINED).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    swa_start = Config.get_swa_start_epoch(
        Config.STUDENT_EPOCHS, Config.STUDENT_SWA_START_RATIO
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=swa_start)

    trainer = Trainer(model, device, criterion, optimizer, scheduler)
    save_path = os.path.join(Config.WORKING_DIR, "student_best.pth")

    # Train Student with SWA
    swa_model = trainer.fit_swa(
        student_loader,
        val_loader,
        total_epochs=Config.STUDENT_EPOCHS,
        swa_start_epoch=swa_start,
        swa_lr=Config.STUDENT_SWA_LR,
        save_path=save_path,
    )

    # ==========================
    # Validation & Analysis
    # ==========================
    print("\n=== Validating Student Model ===")
    val_loss, val_auc = trainer.validate(val_loader, model_to_validate=swa_model)
    print(f"Final Validation Metric: {val_auc}")

    # Failure Analysis
    print("Performing Failure Analysis...")
    swa_model.eval()
    all_errors = []
    all_label_counts = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = swa_model(images)
            probs = torch.sigmoid(outputs)

            # Calculate Mean Absolute Error per sample
            # shape: (batch_size, num_classes) -> mean over classes -> (batch_size,)
            error = torch.abs(probs - labels).mean(dim=1).cpu().numpy()
            all_errors.extend(error)

            # Count number of positive labels per sample
            label_counts = labels.sum(dim=1).cpu().numpy()
            all_label_counts.extend(label_counts)

    if len(all_errors) > 1:
        correlation = np.corrcoef(all_errors, all_label_counts)[0, 1]
        print(f"Correlation between Error and Label Count: {correlation}")
    else:
        print("Not enough samples for correlation analysis.")

    # ==========================
    # Submission
    # ==========================
    THRESHOLD = 0.9433543480067271
    if val_auc > THRESHOLD:
        print(
            f"\nMetric ({val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        swa_model.eval()
        submission_preds = []
        submission_ids = []

        # Predict on Test Set (using test_loader to ensure correct order/ids)
        with torch.no_grad():
            for images, _, rec_ids in test_loader:
                images = images.to(device)
                outputs = swa_model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()

                submission_preds.append(probs)
                submission_ids.append(rec_ids.numpy())

        submission_preds = np.concatenate(submission_preds)
        submission_ids = np.concatenate(submission_ids)

        # Format for CSV: Id,Probability
        # Id = rec_id * 100 + species_id
        output_data = []
        for i, rec_id in enumerate(submission_ids):
            probs = submission_preds[i]
            for species_idx, prob in enumerate(probs):
                row_id = int(rec_id * 100 + species_idx)
                output_data.append({"Id": row_id, "Probability": prob})

        df_submission = pd.DataFrame(output_data)
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nMetric ({val_auc}) <= Threshold ({THRESHOLD}). Submission skipped.")


if __name__ == "__main__":
    main()
