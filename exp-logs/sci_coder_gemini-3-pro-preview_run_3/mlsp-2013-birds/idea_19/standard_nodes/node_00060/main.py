import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    calculate_robust_roc_auc,
)
from library.data import (
    get_fold_loaders,
    prepare_data,
    BirdDataset,
    get_transforms,
    load_image_data,
)
from library.models import get_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from library.engine import train_one_epoch, evaluate


def run_ensemble_inference(model_paths, loader, device, use_tta=False):
    """
    Runs inference using an ensemble of models.
    Supports Horizontal Shift TTA.
    """
    avg_probs = None
    model_count = 0

    for checkpoint_name in model_paths:
        # Determine architecture from filename
        if "resnet18" in checkpoint_name:
            arch = "resnet18"
        elif "efficientnet_b0" in checkpoint_name:
            arch = "efficientnet_b0"
        elif "densenet121" in checkpoint_name:
            arch = "densenet121"
        else:
            continue

        model = get_model(arch, pretrained=False)
        # Load checkpoint
        try:
            load_checkpoint(model, None, checkpoint_name, device)
        except FileNotFoundError:
            print(f"Warning: Checkpoint {checkpoint_name} not found. Skipping.")
            continue

        model.to(device)
        model.eval()

        all_probs = []

        with torch.no_grad():
            for batch in loader:
                # Handle tuple unpacking based on loader return
                if len(batch) == 3:
                    images, _, _ = batch
                else:
                    images, _ = batch

                images = images.to(device)

                if use_tta:
                    # TTA: Original, Left Shift, Right Shift
                    # Shift amount: ~20% of 224 ~= 45 pixels
                    shift = 45

                    # 1. Original
                    out1 = model(images)
                    prob1 = torch.sigmoid(out1)

                    # 2. Left Shift (Roll negative, pad right)
                    img_left = torch.roll(images, shifts=-shift, dims=3)
                    img_left[:, :, :, -shift:] = 0.0
                    out2 = model(img_left)
                    prob2 = torch.sigmoid(out2)

                    # 3. Right Shift (Roll positive, pad left)
                    img_right = torch.roll(images, shifts=shift, dims=3)
                    img_right[:, :, :, :shift] = 0.0
                    out3 = model(img_right)
                    prob3 = torch.sigmoid(out3)

                    # Average TTA
                    batch_probs = (prob1 + prob2 + prob3) / 3.0
                else:
                    # No TTA
                    out = model(images)
                    batch_probs = torch.sigmoid(out)

                all_probs.append(batch_probs.cpu().numpy())

        model_probs = np.concatenate(all_probs, axis=0)

        if avg_probs is None:
            avg_probs = model_probs
        else:
            avg_probs += model_probs

        model_count += 1

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    if model_count == 0:
        return None

    return avg_probs / model_count


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(
        f"Starting training with {len(Config.MODELS)} models, {Config.NUM_FOLDS} folds, {Config.EPOCHS} epochs."
    )

    # 2. Training Loop
    model_paths = []

    for model_name in Config.MODELS:
        print(f"\n=== Training Model Architecture: {model_name} ===")

        for fold in range(Config.NUM_FOLDS):
            print(f"  -- Fold {fold} --")

            # Data Loaders
            train_loader, val_loader = get_fold_loaders(fold, load_cached_data=True)

            # Model
            model = get_model(model_name, pretrained=True)
            model.to(device)

            # Optimizer: AdamW (Cite solution_lesson_node_00059)
            optimizer = AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )

            # Scheduler: Cosine Annealing (Cite solution_lesson_node_00059, solution_lesson_node_00022)
            scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

            # Loss
            criterion = nn.BCEWithLogitsLoss()

            best_auc = -1.0
            best_epoch = -1
            checkpoint_name = f"{model_name}_fold_{fold}_best.pth"

            for epoch in range(Config.EPOCHS):
                # Train
                train_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, device
                )

                # Step Scheduler
                scheduler.step()

                # Validate
                val_loss, val_auc = evaluate(model, val_loader, criterion, device)

                # Checkpoint
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_epoch = epoch
                    save_checkpoint(model, optimizer, epoch, val_auc, checkpoint_name)

                # Logging (minimal to satisfy requirements)
                if (epoch + 1) % 25 == 0:
                    pass  # Suppress verbose logging

            print(f"    Best AUC: {best_auc:.4f} at Epoch {best_epoch}")
            model_paths.append(checkpoint_name)

            # Cleanup
            del model, optimizer, train_loader, val_loader
            torch.cuda.empty_cache()

    # 3. Validation & Failure Analysis
    print("\n=== Performing Validation & Failure Analysis ===")

    # Load the specific hold-out validation set
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    val_images, val_labels, val_ids = load_image_data(
        val_df, Config.SPECTROGRAM_DIR, Config.IMAGE_SIZE
    )

    # Create Dataset/Loader for Validation Set (No Augmentation)
    val_dataset = BirdDataset(
        val_images, val_labels, val_ids, transform=get_transforms(mode="test")
    )
    val_loader_full = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Ensemble Inference
    val_probs = run_ensemble_inference(model_paths, val_loader_full, device)

    if val_probs is not None:
        # Calculate Metric
        final_val_metric = calculate_robust_roc_auc(val_labels, val_probs)
        print(f"Final Validation Metric: {final_val_metric}")

        # Failure Analysis
        # Error: Mean Absolute Error per sample
        errors = np.abs(val_labels - val_probs).mean(axis=1)  # (N,)

        # Input Features: Image Energy and Std
        img_energy = []
        img_std = []
        for img in val_images:
            norm_img = img / 255.0
            img_energy.append(np.mean(norm_img**2))
            img_std.append(np.std(norm_img))

        img_energy = np.array(img_energy)
        img_std = np.array(img_std)

        # Correlations
        if len(errors) > 1:
            corr_energy, _ = pearsonr(errors, img_energy)
            corr_std, _ = pearsonr(errors, img_std)
        else:
            corr_energy, corr_std = 0.0, 0.0

        print(
            f"Failure Analysis - Correlation (Error vs Image Energy): {corr_energy:.4f}"
        )
        print(f"Failure Analysis - Correlation (Error vs Image Std): {corr_std:.4f}")

        # 4. Submission
        THRESHOLD = 0.0
        if final_val_metric > THRESHOLD:
            print("\n=== Generating Submission ===")

            # Load Test Data
            # prepare_data loads test.csv into test_images/test_ids
            _, (test_images, test_ids) = prepare_data(load_cached_data=True)

            test_dataset = BirdDataset(
                test_images, ids=test_ids, transform=get_transforms(mode="test")
            )
            test_loader = DataLoader(
                test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
            )

            # Inference with TTA
            test_probs = run_ensemble_inference(
                model_paths, test_loader, device, use_tta=True
            )

            if test_probs is not None:
                # Format Submission
                submission_rows = []
                for i, rec_id in enumerate(test_ids):
                    probs = test_probs[i]  # (19,)
                    for species_id, prob in enumerate(probs):
                        row_id = int(rec_id * 100 + species_id)
                        submission_rows.append({"Id": row_id, "Probability": prob})

                submission_df = pd.DataFrame(submission_rows)
                submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
                print(f"Submission saved to {Config.SUBMISSION_PATH}")
        else:
            print(
                f"Validation metric {final_val_metric} did not meet threshold {THRESHOLD}. Skipping submission."
            )
    else:
        print("Error: Inference returned no probabilities.")


if __name__ == "__main__":
    run()
