import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.dataset import process_data, WhaleDataset, get_transforms
from library.model import WhaleEfficientNet
from library.loss import WeightedBCELoss
from library.trainer import Trainer
from library.utils import seed_everything, calculate_roc_auc


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("\n--- Data Loading ---")
    # Load Train
    train_specs, train_labels, train_clips = process_data(
        Config.TRAIN_METADATA, "train", load_cached=True
    )
    # Load Val
    val_specs, val_labels, val_clips = process_data(
        Config.VAL_METADATA, "val", load_cached=True
    )
    # Load Test
    test_specs, test_labels, test_clips = process_data(
        Config.TEST_METADATA, "test", load_cached=True
    )

    print(f"Train shape: {train_specs.shape}")
    print(f"Val shape: {val_specs.shape}")
    print(f"Test shape: {test_specs.shape}")

    # 3. Stage 1: Teacher Training
    print("\n--- Stage 1: Teacher Training ---")

    # Datasets & Loaders
    train_dataset = WhaleDataset(
        train_specs, train_labels, transform=get_transforms("train")
    )
    val_dataset = WhaleDataset(val_specs, val_labels, transform=get_transforms("val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model, Loss, Opt
    teacher_model = WhaleEfficientNet(backbone_name=Config.BACKBONE).to(device)
    criterion = WeightedBCELoss(pos_weight=Config.POS_WEIGHT)
    optimizer = AdamW(
        teacher_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Trainer
    teacher_trainer = Trainer(
        teacher_model, criterion, optimizer, scheduler, device, Config
    )

    # Train
    teacher_best_auc = teacher_trainer.fit(
        train_loader, val_loader, save_path=Config.TEACHER_WEIGHTS, epochs=Config.EPOCHS
    )
    print(f"Teacher Best AUC: {teacher_best_auc}")

    # 4. Stage 2: Pseudo-Labeling
    print("\n--- Stage 2: Pseudo-Labeling ---")

    # Load best teacher
    teacher_model.load_state_dict(
        torch.load(Config.TEACHER_WEIGHTS, map_location=device)
    )
    teacher_model.eval()

    # Predict on Test
    test_dataset_for_pred = WhaleDataset(
        test_specs, np.zeros(len(test_specs)), is_test=True
    )
    test_loader = DataLoader(
        test_dataset_for_pred,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_probs = teacher_trainer.predict(test_loader)

    # Combine Data
    # Specs: Train + Test
    combined_specs = np.concatenate([train_specs, test_specs], axis=0)

    # Labels (for Dataset init, though pseudo_labels will override)
    # We just need an array of correct length
    combined_labels_dummy = np.concatenate(
        [train_labels, np.zeros(len(test_specs))], axis=0
    )

    # Pseudo Labels: Train (Hard) + Test (Soft)
    train_labels_float = train_labels.astype(np.float32)
    test_probs_float = test_probs.astype(np.float32)
    combined_pseudo_labels = np.concatenate(
        [train_labels_float, test_probs_float], axis=0
    )

    print(f"Combined Dataset Size: {len(combined_specs)}")

    # 5. Stage 3: Student Training
    print("\n--- Stage 3: Student Training ---")

    # Student Dataset
    student_dataset = WhaleDataset(
        combined_specs,
        combined_labels_dummy,
        transform=get_transforms("train"),
        pseudo_labels=combined_pseudo_labels,
    )

    student_loader = DataLoader(
        student_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Student Model
    student_model = WhaleEfficientNet(backbone_name=Config.BACKBONE).to(device)
    optimizer_student = AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler_student = CosineAnnealingLR(optimizer_student, T_max=Config.EPOCHS)

    # Student Trainer
    student_trainer = Trainer(
        student_model, criterion, optimizer_student, scheduler_student, device, Config
    )

    # Train Student
    student_best_auc = student_trainer.fit(
        student_loader,
        val_loader,
        save_path=Config.STUDENT_WEIGHTS,
        epochs=Config.EPOCHS,
    )
    print(f"Student Best AUC: {student_best_auc}")

    # 6. Validation & Analysis
    print("\n--- Validation & Failure Analysis ---")

    # Ensure best student weights are loaded
    student_model.load_state_dict(
        torch.load(Config.STUDENT_WEIGHTS, map_location=device)
    )

    # Get predictions on Validation Set
    val_probs = student_trainer.predict(val_loader)

    # Compute Metric
    final_metric = calculate_roc_auc(val_labels, val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate Error
    errors = np.abs(val_labels - val_probs)

    # Calculate Input Features (Mean, Std of Spectrograms)
    # val_specs shape: (N, F, T)
    spec_means = val_specs.mean(axis=(1, 2))
    spec_stds = val_specs.std(axis=(1, 2))

    analysis_df = pd.DataFrame(
        {"error": errors, "spec_mean": spec_means, "spec_std": spec_stds}
    )

    correlations = analysis_df.corr()["error"]
    print("\nCorrelation of Error with Input Features:")
    print(correlations)

    # 7. Submission
    threshold = 0.9960914834372254
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )

        # Predict on Test Set
        final_test_probs = student_trainer.predict(test_loader)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"clip": test_clips, "probability": final_test_probs}
        )

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
