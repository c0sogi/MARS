import os
import pandas as pd
import numpy as np
import soundfile as sf
import torch
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.model import WhaleModel
from library.dataset import get_dataloaders
from library.trainer import Trainer, get_pos_weight, generate_submission

# --- Configuration Overrides for Fast Baseline ---
# Removed overrides to allow full training duration (25 epochs) as per task allowance.
# Config.EPOCHS will default to 25.


def run_failure_analysis(model, val_loader, val_metadata_path):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and input features.
    """
    print("\n--- Starting Failure Analysis ---")

    # 1. Get Model Predictions
    model.eval()
    all_preds = []
    all_targets = []

    device = Config.DEVICE
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 2. Calculate Error
    errors = np.abs(all_targets - all_preds)

    # 3. Extract Features from Audio Files
    # We read the validation metadata to locate files.
    # Note: The DataLoader with shuffle=False preserves the order of the metadata CSV.
    val_df = pd.read_csv(val_metadata_path)

    durations = []
    mean_amps = []
    std_amps = []

    print("Extracting audio features for correlation analysis...")
    for _, row in val_df.iterrows():
        file_path = os.path.join(Config.INPUT_ROOT, row["file_path"])
        try:
            audio, sr = sf.read(file_path)
            durations.append(len(audio) / sr)
            mean_amps.append(np.mean(np.abs(audio)))
            std_amps.append(np.std(audio))
        except Exception:
            # Fallback for corrupt files (though metadata check passed)
            durations.append(Config.DURATION)
            mean_amps.append(0)
            std_amps.append(0)

    # 4. Calculate Correlations
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "duration": durations,
            "mean_amp": mean_amps,
            "std_amp": std_amps,
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")

    print("\nCorrelation between Model Error and Input Features:")
    print(correlations)
    return correlations


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.init_directories()

    print(f"Running on device: {Config.DEVICE}")

    # ---------------------------------------------------------
    # Phase 1: Teacher Training
    # ---------------------------------------------------------
    print("\n=== Phase 1: Teacher Training ===")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, config=Config
    )

    # Initialize Teacher Model
    teacher_model = WhaleModel(config=Config, pretrained=True)

    # Calculate Class Weight
    # We access the underlying dataset to calculate weights
    pos_weight = get_pos_weight(train_loader.dataset)
    print(f"Positive Class Weight: {pos_weight:.4f}")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        teacher_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LEARNING_RATE
    )

    # Trainer
    teacher_trainer = Trainer(
        teacher_model, optimizer, scheduler, device=Config.DEVICE, pos_weight=pos_weight
    )

    # Train Teacher (No Mixup for Teacher to ensure clean baseline)
    teacher_auc = teacher_trainer.fit(
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        use_mixup=False,
        save_name="teacher_best.pth",
    )
    print(f"Best Teacher AUC: {teacher_auc}")

    # ---------------------------------------------------------
    # Phase 2: Pseudo-Labeling
    # ---------------------------------------------------------
    print("\n=== Phase 2: Pseudo-Labeling ===")

    # Reload Best Teacher Weights
    print("Reloading best teacher weights...")
    load_checkpoint(teacher_model, filename="teacher_best.pth", device=Config.DEVICE)

    # Generate Soft Labels
    soft_pseudo_labels = teacher_trainer.predict(test_loader)
    print(f"Generated {len(soft_pseudo_labels)} pseudo-labels.")

    # ---------------------------------------------------------
    # Phase 3: Student Training (Noisy Student)
    # ---------------------------------------------------------
    print("\n=== Phase 3: Student Training ===")

    # Get Augmented Dataloaders (Train + Pseudo-labeled Test)
    # Note: We reload dataloaders to include pseudo-labels
    student_train_loader, student_val_loader, student_test_loader = get_dataloaders(
        load_cached_data=True, config=Config, pseudo_labels=soft_pseudo_labels
    )

    # Initialize Student Model (Fresh initialization)
    student_model = WhaleModel(config=Config, pretrained=True)

    # Optimizer & Scheduler for Student
    optimizer_student = torch.optim.AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler_student = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_student, T_max=Config.EPOCHS, eta_min=Config.MIN_LEARNING_RATE
    )

    # Trainer for Student
    student_trainer = Trainer(
        student_model,
        optimizer_student,
        scheduler_student,
        device=Config.DEVICE,
        pos_weight=pos_weight,  # Reuse weight or recalculate if needed, but keeping it stable is fine
    )

    # Train Student (With Mixup for Noise Injection)
    student_auc = student_trainer.fit(
        student_train_loader,
        student_val_loader,
        epochs=Config.EPOCHS,
        use_mixup=True,
        save_name="student_best.pth",
    )

    # ---------------------------------------------------------
    # Evaluation & Failure Analysis
    # ---------------------------------------------------------
    print("\n=== Final Evaluation ===")

    # Reload Best Student Weights
    load_checkpoint(student_model, filename="student_best.pth", device=Config.DEVICE)

    # Validate
    val_loss, final_val_auc = student_trainer.validate(student_val_loader)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_val_auc}")

    # Failure Analysis
    run_failure_analysis(student_model, student_val_loader, Config.VAL_METADATA_PATH)

    # ---------------------------------------------------------
    # Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.9960914834372254

    if final_val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        predictions = student_trainer.predict(student_test_loader)
        generate_submission(predictions, Config.SUBMISSION_PATH)

    else:
        print(
            f"\nValidation AUC ({final_val_auc}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
