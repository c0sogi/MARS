import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, compute_multilabel_auc
from library.data import get_dataloaders
from library.model import get_model
from library.engine import train_one_epoch, validate, SWAHandler, inference


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.get_device()

    print(f"Starting Heterogeneous Ensemble Distillation on {device}")

    # -------------------------------------------------------------------------
    # Stage 1: Train Teachers
    # -------------------------------------------------------------------------
    print("\n--- Stage 1: Training Heterogeneous Teacher Ensemble ---")

    # Load initial data (only labeled train data)
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, load_cached_data=True
    )

    teacher_test_preds = []

    for i, t_conf in enumerate(Config.TEACHER_MODELS):
        model_name = t_conf["arch"]
        model_id = t_conf["id"]
        print(
            f"\nTraining Teacher {i+1}/{len(Config.TEACHER_MODELS)}: {model_id} ({model_name})"
        )

        # Init Model
        model = get_model(model_name, Config.NUM_CLASSES, pretrained=True)
        model.to(device)

        # Init Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Init SWA
        swa_handler = SWAHandler(model, Config.TEACHER_SWA_START_EPOCH, device)

        # Training Loop
        for epoch in range(Config.TEACHER_EPOCHS):
            # Adjust LR for SWA if active
            if epoch >= Config.TEACHER_SWA_START_EPOCH:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = Config.TEACHER_SWA_LR

            loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
            swa_handler.on_epoch_end(model, epoch)

            # Optional: minimal logging
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{Config.TEACHER_EPOCHS} - Loss: {loss:.4f}")

        # Finalize SWA
        swa_handler.finalize(train_loader)
        final_teacher = swa_handler.get_model() if swa_handler.active else model

        # Inference on Test Set (for Pseudo-Labeling)
        print(f"  Generating predictions for {model_id}...")
        preds = inference(final_teacher, test_loader, device, use_tta=True)
        teacher_test_preds.append(preds)

        # Clear memory
        del model, final_teacher, optimizer, swa_handler
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Stage 2: Robust Pseudo-Labeling
    # -------------------------------------------------------------------------
    print("\n--- Stage 2: Generating Pseudo-Labels ---")

    # Average predictions
    avg_preds = np.mean(teacher_test_preds, axis=0)

    # Create Pseudo-Label DataFrame
    # We need rec_ids from the test set to map predictions correctly
    test_rec_ids = test_loader.dataset.df["rec_id"].values

    pseudo_data = {"rec_id": test_rec_ids}
    for k in range(Config.NUM_CLASSES):
        pseudo_data[f"species_{k}"] = avg_preds[:, k]

    df_pseudo = pd.DataFrame(pseudo_data)

    pseudo_path = os.path.join(Config.WORKING_DIR, "pseudo_labels.parquet")
    df_pseudo.to_parquet(pseudo_path, index=False)
    print(f"Pseudo-labels saved to {pseudo_path}")

    # -------------------------------------------------------------------------
    # Stage 3: Student Training
    # -------------------------------------------------------------------------
    print("\n--- Stage 3: Training Student Model (Distillation) ---")

    # Reload DataLoaders with Pseudo-Labels merged into Train
    # Note: load_cached_data=False forces re-processing with the new pseudo-labels
    train_loader_student, val_loader_student, test_loader_student = get_dataloaders(
        debug=Config.DEBUG, pseudo_labels_path=pseudo_path, load_cached_data=False
    )

    print(f"Student Training Set Size: {len(train_loader_student.dataset)}")

    # Init Student Model
    student_model = get_model(Config.STUDENT_ARCH, Config.NUM_CLASSES, pretrained=True)
    student_model.to(device)

    # Init Optimizer
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Init SWA
    swa_handler = SWAHandler(student_model, Config.STUDENT_SWA_START_EPOCH, device)

    # Training Loop
    for epoch in range(Config.STUDENT_EPOCHS):
        if epoch >= Config.STUDENT_SWA_START_EPOCH:
            for param_group in optimizer.param_groups:
                param_group["lr"] = Config.STUDENT_SWA_LR

        loss = train_one_epoch(
            student_model, train_loader_student, optimizer, device, epoch
        )
        swa_handler.on_epoch_end(student_model, epoch)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{Config.STUDENT_EPOCHS} - Loss: {loss:.4f}")

    # Finalize SWA
    swa_handler.finalize(train_loader_student)
    final_student = swa_handler.get_model() if swa_handler.active else student_model

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    print("\n--- Evaluation ---")

    # Validate
    val_loss, val_auc = validate(final_student, val_loader_student, device)

    # REQUIRED PRINT
    print(f"Final Validation Metric: {val_auc}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    final_student.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader_student:
            images = images.to(device)
            outputs = final_student(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate per-sample error (Mean Absolute Error)
    per_sample_mae = np.mean(np.abs(all_targets - all_preds), axis=1)

    # Feature: Label Cardinality (Number of species present)
    cardinality = np.sum(all_targets, axis=1)

    # Correlation
    if len(per_sample_mae) > 1:
        correlation = np.corrcoef(per_sample_mae, cardinality)[0, 1]
        print(
            f"Correlation between Error Magnitude and Label Cardinality: {correlation:.4f}"
        )
    else:
        print("Not enough samples for correlation analysis.")

    # -------------------------------------------------------------------------
    # Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.9594082190886809

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference on Test Set
        test_probs = inference(
            final_student, test_loader_student, device, use_tta=False
        )

        # Format Submission
        # Id = rec_id * 100 + species_id
        submission_rows = []
        test_rec_ids = test_loader_student.dataset.df["rec_id"].values

        for idx, rec_id in enumerate(test_rec_ids):
            probs = test_probs[idx]
            for species_id, prob in enumerate(probs):
                row_id = int(rec_id * 100 + species_id)
                submission_rows.append({"Id": row_id, "Probability": prob})

        submission_df = pd.DataFrame(submission_rows)

        # Sort by Id just in case
        submission_df = submission_df.sort_values("Id")

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({val_auc}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
