import os
import sys
import numpy as np
import torch
import pandas as pd
import gc

# Import from the provided library
from library.config import Config
from library.utils import set_seed, load_checkpoint, calculate_roc_auc
from library.dataset import get_dataloaders
from library.model import get_model
from library.training import run_training
from library.inference import run_inference, generate_pseudo_labels, predict_and_submit


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)

    print("============================================================")
    print("   Multi-Generational High-Fidelity ResNet-34 Distillation  ")
    print("============================================================")

    # Define checkpoint directories
    teacher_ckpt_base = os.path.join(Config.CHECKPOINT_DIR, "teachers")
    student1_ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, "student_1")
    student2_ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, "student_2")

    # ------------------------------------------------------------------
    # Stage 1: Train Teacher Ensemble (Generation 0)
    # ------------------------------------------------------------------
    print("\n--- Stage 1: Training Teacher Ensemble (3 Models) ---")

    teacher_preds = []
    test_rec_ids = None

    # Train 3 independent teachers
    for i in range(3):
        print(f"\nTraining Teacher {i}...")

        # Vary seed for independence
        iter_seed = Config.SEED + i
        ckpt_dir = os.path.join(teacher_ckpt_base, f"fold_{i}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Train
        best_auc, swa_auc = run_training(
            pseudo_labels_df=None,
            checkpoint_dir=ckpt_dir,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            seed=iter_seed,
        )

        print(
            f"Teacher {i} Results -> Best AUC: {best_auc:.6f} | SWA AUC: {swa_auc:.6f}"
        )

        # Inference on Test Set for Pseudo-Labeling
        # Load SWA model
        model = get_model(
            pretrained=False, num_classes=Config.NUM_CLASSES, device=Config.DEVICE
        )
        load_checkpoint(
            model, os.path.join(ckpt_dir, "model_swa.pth"), device=Config.DEVICE
        )

        # Get Test Loader
        _, _, test_loader = get_dataloaders(
            train_metadata=Config.TRAIN_METADATA,
            val_metadata=Config.VAL_METADATA,
            test_metadata=Config.TEST_METADATA,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
        )

        # Run Inference with TTA
        ids, probs = run_inference(
            model, test_loader, device=Config.DEVICE, use_tta=True
        )

        teacher_preds.append(probs)
        if test_rec_ids is None:
            test_rec_ids = ids

        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Ensemble Predictions (Average)
    avg_teacher_probs = np.mean(teacher_preds, axis=0)

    # Generate Pseudo-Labels (Soft Labels)
    pseudo_labels_v1 = generate_pseudo_labels(
        test_rec_ids, avg_teacher_probs, threshold=None
    )
    print(f"\nGenerated {len(pseudo_labels_v1)} pseudo-labels from Teacher Ensemble.")

    # ------------------------------------------------------------------
    # Stage 2: Train Student 1 (Generation 1)
    # ------------------------------------------------------------------
    print("\n--- Stage 2: Training Student 1 (Generation 1) ---")

    os.makedirs(student1_ckpt_dir, exist_ok=True)
    student1_seed = Config.SEED + 100

    # Train Student 1 on Combined Data (Train + Pseudo V1)
    best_auc_s1, swa_auc_s1 = run_training(
        pseudo_labels_df=pseudo_labels_v1,
        checkpoint_dir=student1_ckpt_dir,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        seed=student1_seed,
    )

    print(
        f"Student 1 Results -> Best AUC: {best_auc_s1:.6f} | SWA AUC: {swa_auc_s1:.6f}"
    )

    # Refine Pseudo-Labels using Student 1 (SWA)
    print("Refining pseudo-labels with Student 1...")
    model_s1 = get_model(
        pretrained=False, num_classes=Config.NUM_CLASSES, device=Config.DEVICE
    )
    load_checkpoint(
        model_s1, os.path.join(student1_ckpt_dir, "model_swa.pth"), device=Config.DEVICE
    )

    _, _, test_loader = get_dataloaders(
        train_metadata=Config.TRAIN_METADATA,
        val_metadata=Config.VAL_METADATA,
        test_metadata=Config.TEST_METADATA,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    ids_s1, probs_s1 = run_inference(
        model_s1, test_loader, device=Config.DEVICE, use_tta=True
    )

    pseudo_labels_v2 = generate_pseudo_labels(ids_s1, probs_s1, threshold=None)

    del model_s1
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------
    # Stage 3: Train Student 2 (Generation 2)
    # ------------------------------------------------------------------
    print("\n--- Stage 3: Training Student 2 (Generation 2) ---")

    os.makedirs(student2_ckpt_dir, exist_ok=True)
    student2_seed = Config.SEED + 200

    # Train Student 2 on Combined Data (Train + Pseudo V2)
    best_auc_s2, swa_auc_s2 = run_training(
        pseudo_labels_df=pseudo_labels_v2,
        checkpoint_dir=student2_ckpt_dir,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        seed=student2_seed,
    )

    # ------------------------------------------------------------------
    # Final Evaluation & Failure Analysis
    # ------------------------------------------------------------------
    print("\n--- Final Evaluation ---")

    # Print the required metric format
    print(f"Final Validation Metric: {swa_auc_s2}")

    # Load the final model for analysis
    final_model_path = os.path.join(student2_ckpt_dir, "model_swa.pth")
    model_final = get_model(
        pretrained=False, num_classes=Config.NUM_CLASSES, device=Config.DEVICE
    )
    load_checkpoint(model_final, final_model_path, device=Config.DEVICE)
    model_final.eval()

    # Failure Analysis on Validation Set
    print("\nPerforming Failure Analysis...")
    _, val_loader, _ = get_dataloaders(
        train_metadata=Config.TRAIN_METADATA,
        val_metadata=Config.VAL_METADATA,
        test_metadata=Config.TEST_METADATA,
        batch_size=1,  # Process one by one for granular analysis
        num_workers=0,
    )

    errors = []
    img_means = []
    img_stds = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            outputs = model_final(images)
            probs = torch.sigmoid(outputs)

            # Calculate Mean Absolute Error for this sample
            mae = torch.abs(labels - probs).mean().item()
            errors.append(mae)

            # Calculate Image Statistics (using first channel of the RGB replica)
            # images is [1, 3, H, W]
            img_np = images[0, 0].cpu().numpy()
            img_means.append(np.mean(img_np))
            img_stds.append(np.std(img_np))

    # Calculate Correlations
    if len(errors) > 1:
        corr_mean = np.corrcoef(errors, img_means)[0, 1]
        corr_std = np.corrcoef(errors, img_stds)[0, 1]

        print(f"Correlation (Model Error vs Input Mean Intensity): {corr_mean:.6f}")
        print(f"Correlation (Model Error vs Input Contrast/Std): {corr_std:.6f}")
    else:
        print("Not enough validation samples for correlation analysis.")

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    # Threshold check as per task description
    THRESHOLD_METRIC = 0.9594082190886809

    if swa_auc_s2 > THRESHOLD_METRIC:
        print(
            f"\nValidation Metric ({swa_auc_s2}) > Threshold ({THRESHOLD_METRIC}). Generating Submission..."
        )
        predict_and_submit(
            final_model_path, output_path=Config.SUBMISSION_PATH, use_tta=True
        )
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation Metric ({swa_auc_s2}) <= Threshold ({THRESHOLD_METRIC}). Skipping Submission."
        )

    print("\nRun Complete.")


if __name__ == "__main__":
    main()
