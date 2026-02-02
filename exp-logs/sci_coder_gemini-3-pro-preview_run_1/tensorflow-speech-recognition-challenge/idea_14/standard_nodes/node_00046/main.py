import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import DilatedEfficientNet
from library.trainer import Trainer


def run():
    # 1. Configuration & Setup
    # Increase epochs to ensure convergence (Cite Lesson 3)
    Config.epochs_teacher = 25
    Config.epochs_student = 25
    Config.debug = False
    Config.num_workers = 8

    # Set seeds
    set_seed(Config.seed)
    Config.create_dirs()

    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader, label_encoder = get_dataloaders(
        load_cached_data=True
    )

    # Initialize Trainer helper (we will use its methods for training steps)
    trainer = Trainer(
        train_loader, val_loader, test_loader, label_encoder, config=Config
    )
    device = Config.device

    # ==========================================
    # Stage 1: Train Teacher
    # ==========================================
    print("\nStarting Stage 1: Teacher Training")
    teacher_model = DilatedEfficientNet(config=Config).to(device)

    optimizer_teacher = optim.AdamW(
        teacher_model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
    )
    scheduler_teacher = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_teacher, T_max=Config.epochs_teacher, eta_min=Config.min_lr
    )

    best_teacher_acc = 0.0
    best_teacher_path = os.path.join(Config.checkpoint_dir, "teacher_best.pth")

    for epoch in range(Config.epochs_teacher):
        train_loss, train_acc = trainer.train_teacher_epoch(
            teacher_model, optimizer_teacher
        )
        val_loss, val_acc = trainer.validate(teacher_model)
        scheduler_teacher.step()

        print(
            f"Teacher Epoch {epoch+1}/{Config.epochs_teacher} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_teacher_acc:
            best_teacher_acc = val_acc
            torch.save(teacher_model.state_dict(), best_teacher_path)

    print(f"Stage 1 Complete. Best Teacher Acc: {best_teacher_acc:.6f}")

    # ==========================================
    # Stage 2: Train Student (Distillation)
    # ==========================================
    print("\nStarting Stage 2: Student Distillation")

    # Load Best Teacher and Freeze
    teacher_model.load_state_dict(torch.load(best_teacher_path, map_location=device))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Initialize Fresh Student
    student_model = DilatedEfficientNet(config=Config).to(device)

    optimizer_student = optim.AdamW(
        student_model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
    )
    scheduler_student = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_student, T_max=Config.epochs_student, eta_min=Config.min_lr
    )

    best_student_acc = 0.0
    best_student_path = os.path.join(Config.checkpoint_dir, "student_best.pth")

    for epoch in range(Config.epochs_student):
        train_loss, train_acc = trainer.train_student_epoch(
            student_model, teacher_model, optimizer_student
        )
        val_loss, val_acc = trainer.validate(student_model)
        scheduler_student.step()

        print(
            f"Student Epoch {epoch+1}/{Config.epochs_student} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_student_acc:
            best_student_acc = val_acc
            torch.save(student_model.state_dict(), best_student_path)

    print(f"Stage 2 Complete. Best Student Acc: {best_student_acc:.6f}")

    # ==========================================
    # Final Validation & Failure Analysis
    # ==========================================
    print("\nPerforming Final Validation and Failure Analysis...")

    # Load Best Student
    student_model.load_state_dict(torch.load(best_student_path, map_location=device))
    student_model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = student_model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Metric
    # Map fine-grained predictions to 12-class competition schema for final validation (Cite Lesson 31)
    pred_labels = label_encoder.inverse_transform(all_preds)
    target_labels = label_encoder.inverse_transform(all_targets)

    def map_to_12(label):
        if label in Config.target_labels:
            return label
        if label == Config.silence_label:
            return label
        return Config.unknown_label

    mapped_preds = np.array([map_to_12(l) for l in pred_labels])
    mapped_targets = np.array([map_to_12(l) for l in target_labels])

    final_accuracy = np.mean(mapped_preds == mapped_targets)
    print(f"Final Validation Metric (12-class): {final_accuracy}")

    # Failure Analysis
    # Construct DataFrame for analysis
    val_df = val_loader.dataset.df.copy()

    # Ensure lengths match (loader might drop last if drop_last=True, but val loader usually doesn't)
    # The provided val_loader in dataset.py does NOT drop last.
    if len(val_df) != len(all_preds):
        print(
            "Warning: Validation dataframe length mismatch. Truncating to match predictions."
        )
        val_df = val_df.iloc[: len(all_preds)]

    val_df["pred"] = all_preds
    val_df["target"] = all_targets
    val_df["error_magnitude"] = (val_df["pred"] != val_df["target"]).astype(int)

    # Feature 1: Target Label Index
    val_df["label_idx"] = val_df["target"]

    # Feature 2: File Size (Proxy for duration/info content)
    # We use os.path.getsize. Prepend input_dir.
    val_df["file_size"] = val_df["filepath"].apply(
        lambda x: os.path.getsize(os.path.join(Config.input_dir, x))
    )

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    correlations = val_df[["error_magnitude", "label_idx", "file_size"]].corr()[
        "error_magnitude"
    ]
    print(correlations)

    # ==========================================
    # Conditional Submission
    # ==========================================
    threshold = 0.9872909698996656

    if final_accuracy > threshold:
        print(
            f"\nMetric ({final_accuracy}) > Threshold ({threshold}). Generating submission..."
        )
        trainer.generate_submission(best_student_path)
    else:
        print(
            f"\nMetric ({final_accuracy}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
