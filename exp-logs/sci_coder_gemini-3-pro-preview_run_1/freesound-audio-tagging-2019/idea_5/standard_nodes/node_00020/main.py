import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import soundfile as sf

# Import from library
from library.config import Config
from library.utils import set_seed, get_logger
from library.dataset import get_dataloaders, AudioDataset, collate_fn
from library.model import AudioEfficientNet
from library.engine import (
    Trainer,
    get_or_compute_teacher_predictions,
    generate_submission,
    validate,
)


def run_failure_analysis(model, val_loader, device, val_df):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and metadata features.
    """
    model.eval()
    all_targets = []
    all_scores = []

    # Get predictions
    with torch.no_grad():
        for data, target, _ in val_loader:
            data = data.to(device)
            output = model(data)
            scores = torch.sigmoid(output)
            all_targets.append(target.numpy())
            all_scores.append(scores.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_scores = np.concatenate(all_scores)

    # Calculate error per sample (Mean Absolute Error per sample across classes)
    errors = np.abs(all_targets - all_scores).mean(axis=1)

    # Add errors to dataframe
    val_df = val_df.copy()
    val_df["error"] = errors

    # Compute label_count from the 'labels' string column
    val_df["label_count"] = val_df["labels"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) else 0
    )

    # extract duration for correlation analysis
    durations = []
    for idx, row in val_df.iterrows():
        try:
            path = os.path.join(Config.INPUT_ROOT, row["filepath"])
            # Read only header for speed
            info = sf.info(path)
            durations.append(info.duration)
        except Exception:
            durations.append(0.0)
    val_df["duration"] = durations

    # Correlations
    corr_duration = val_df["duration"].corr(val_df["error"])
    corr_labels = val_df["label_count"].corr(val_df["error"])

    print(f"Correlation between Error and Duration: {corr_duration:.4f}")
    print(f"Correlation between Error and Label Count: {corr_labels:.4f}")


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    logger = get_logger("main")

    # Override Config for Fast Baseline execution
    # Reducing epochs to ensure completion within the 2-hour limit
    Config.EPOCHS = 12

    logger.info(f"Starting execution with Device: {Config.DEVICE}")

    # 2. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 3. Teacher Training
    logger.info("=== Stage 1: Teacher Training ===")

    # Teacher DataLoaders (No soft labels yet)
    train_loader_teacher, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, soft_labels=None
    )

    # Model
    teacher_model = AudioEfficientNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        teacher_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    # Trainer
    teacher_trainer = Trainer(
        model=teacher_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        save_path=Config.TEACHER_MODEL_PATH,
        patience=5,
    )

    # Train Teacher
    teacher_trainer.fit(train_loader_teacher, val_loader, epochs=Config.EPOCHS)

    # Load best teacher weights
    teacher_model.load_state_dict(
        torch.load(Config.TEACHER_MODEL_PATH, map_location=Config.DEVICE)
    )

    # 4. Pseudo-Labeling (Distillation)
    logger.info("=== Stage 2: Pseudo-Labeling ===")

    # Identify Noisy Data
    noisy_df = train_df[train_df["filepath"].str.contains("train_noisy")].reset_index(
        drop=True
    )
    logger.info(f"Generating soft labels for {len(noisy_df)} noisy samples.")

    # Create a specific loader for noisy data (inference mode: full length/no shuffle)
    noisy_ds = AudioDataset(noisy_df, mode="val")
    noisy_loader = torch.utils.data.DataLoader(
        noisy_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Generate/Load Predictions
    soft_labels = get_or_compute_teacher_predictions(
        teacher_model,
        noisy_loader,
        Config.DEVICE,
        Config.TEACHER_PREDS_NPY,
        load_cached_data=True,
    )

    # 5. Student Training
    logger.info("=== Stage 3: Student Training ===")

    # Student DataLoaders (With soft labels)
    train_loader_student, _, _ = get_dataloaders(
        train_df, val_df, test_df, soft_labels=soft_labels
    )

    # Student Model (Fresh init)
    student_model = AudioEfficientNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)

    # Optimizer & Scheduler for Student
    optimizer_student = optim.AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler_student = CosineAnnealingLR(
        optimizer_student, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # Trainer
    student_trainer = Trainer(
        model=student_model,
        optimizer=optimizer_student,
        scheduler=scheduler_student,
        device=Config.DEVICE,
        save_path=Config.STUDENT_MODEL_PATH,
        patience=5,
    )

    # Train Student
    student_trainer.fit(train_loader_student, val_loader, epochs=Config.EPOCHS)

    # Load best student weights
    student_model.load_state_dict(
        torch.load(Config.STUDENT_MODEL_PATH, map_location=Config.DEVICE)
    )

    # 6. Validation & Analysis
    logger.info("=== Stage 4: Evaluation & Analysis ===")

    val_loss, val_score = validate(student_model, val_loader, Config.DEVICE)
    print(f"Final Validation Metric: {val_score}")

    logger.info("Running Failure Analysis...")
    run_failure_analysis(student_model, val_loader, Config.DEVICE, val_df)

    # 7. Submission
    threshold = 0.8554930465617762
    if val_score > threshold:
        logger.info(
            f"Validation score {val_score} exceeds threshold {threshold}. Generating submission."
        )
        generate_submission(
            student_model, test_loader, Config.DEVICE, Config.SUBMISSION_PATH
        )
    else:
        logger.info(
            f"Validation score {val_score} does not exceed threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
