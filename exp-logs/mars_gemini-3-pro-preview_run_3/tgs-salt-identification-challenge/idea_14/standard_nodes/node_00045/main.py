import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_iou_batch, rle_encode
from library.dataset import get_loaders, get_test_loader, SaltDataset, get_transforms
from library.model import SaltUNetPlusPlus
from library.engine import SaltEngine
from library.inference import predict_with_tta, optimize_threshold, generate_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("Initializing Fast Baseline Run...")
    seed_everything(Config.SEED)

    # Fast Baseline Overrides
    PHASE1_EPOCHS = 15
    PHASE2_EPOCHS = 5
    FOLD_TO_RUN = 0

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Phase 1: Supervised Training (Fold 0)
    # -------------------------------------------------------------------------
    print(f"\n--- Phase 1: Supervised Training (Fold {FOLD_TO_RUN}) ---")

    # Get Loaders
    train_loader, val_loader = get_loaders(fold=FOLD_TO_RUN, load_cached_data=True)

    # Initialize Model
    model = SaltUNetPlusPlus(
        encoder_name=Config.ENCODER,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=Config.CHANNELS,
        classes=1,
    ).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # Engine
    engine = SaltEngine(model, device, optimizer, scheduler)

    # Train Phase 1
    phase1_ckpt = os.path.join(Config.CHECKPOINT_DIR, f"fold_{FOLD_TO_RUN}_phase1.pth")
    engine.fit(
        train_loader,
        val_loader,
        epochs=PHASE1_EPOCHS,
        save_path=phase1_ckpt,
        phase2=False,
        patience=5,
    )

    # Load Best Phase 1 Weights
    print("Loading best Phase 1 model for pseudo-labeling...")
    model.load_state_dict(torch.load(phase1_ckpt, map_location=device))

    # -------------------------------------------------------------------------
    # 3. Pseudo-Label Generation
    # -------------------------------------------------------------------------
    print("\n--- Generating Pseudo-Labels for Test Set ---")
    test_loader = get_test_loader(load_cached_data=True)

    # Predict on Test Set (Returns Soft Probabilities)
    test_results = predict_with_tta(model, test_loader, device)
    soft_pseudo_labels = test_results["preds"]  # (N_test, 101, 101)

    # -------------------------------------------------------------------------
    # 4. Phase 2: Semi-Supervised Self-Training
    # -------------------------------------------------------------------------
    print(f"\n--- Phase 2: Semi-Supervised Training (Fold {FOLD_TO_RUN}) ---")

    # Construct Combined Dataset
    # 1. Retrieve original training data arrays
    train_ds = train_loader.dataset
    train_images = train_ds.images
    train_masks = train_ds.masks.astype(np.float32)  # Hard masks as float
    train_depths = train_ds.depths
    train_ids = train_ds.ids

    # 2. Retrieve test data arrays
    test_ds = test_loader.dataset
    test_images = test_ds.images
    test_depths = test_ds.depths
    test_ids = test_ds.ids

    # 3. Concatenate
    combined_images = np.concatenate([train_images, test_images], axis=0)
    combined_masks = np.concatenate([train_masks, soft_pseudo_labels], axis=0)
    combined_depths = np.concatenate([train_depths, test_depths], axis=0)
    combined_ids = np.concatenate([train_ids, test_ids], axis=0)

    # 4. Create Combined Dataset
    combined_dataset = SaltDataset(
        images=combined_images,
        masks=None,  # We use pseudo_labels argument for unified handling
        depths=combined_depths,
        ids=combined_ids,
        phase="train",
        transform=get_transforms("train"),
        pseudo_labels=combined_masks,
    )

    combined_loader = DataLoader(
        combined_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Re-init Optimizer for Fine-tuning (Lower LR)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE * 0.5,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    engine = SaltEngine(model, device, optimizer, scheduler)

    # Train Phase 2
    phase2_ckpt = os.path.join(Config.CHECKPOINT_DIR, f"fold_{FOLD_TO_RUN}_best.pth")
    engine.fit(
        combined_loader,
        val_loader,
        epochs=PHASE2_EPOCHS,
        save_path=phase2_ckpt,
        phase2=True,
        patience=3,
    )

    # Load Best Phase 2 Weights
    print("Loading best Phase 2 model for final evaluation...")
    model.load_state_dict(torch.load(phase2_ckpt, map_location=device))

    # -------------------------------------------------------------------------
    # 5. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Final Validation & Failure Analysis ---")

    # Run Inference on Validation Set
    val_results = predict_with_tta(model, val_loader, device)
    val_preds = val_results["preds"]
    val_targets = val_results["targets"]

    # Calculate Final Metric
    final_metric = calculate_iou_batch(val_preds, val_targets, threshold=0.5)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis
    print("Performing Failure Analysis...")

    # Calculate per-image mAP (Error = 1 - mAP)
    errors = []
    # We need to replicate the metric calculation logic per image
    iou_thresholds = np.arange(0.5, 0.96, 0.05)

    for i in range(len(val_preds)):
        p = (val_preds[i] > 0.5).astype(np.uint8)
        t = (val_targets[i] > 0.5).astype(np.uint8)

        intersection = np.sum(p & t)
        union = np.sum(p | t)
        iou = 1.0 if union == 0 else intersection / union

        matches = iou > iou_thresholds
        score = np.mean(matches)
        errors.append(1.0 - score)

    errors = np.array(errors)

    # Extract Metadata Features
    val_ds = val_loader.dataset
    depths = val_ds.depths

    # Calculate Salt Coverage (Target)
    coverages = np.array([np.mean(t) for t in val_targets])

    # Calculate Image Intensity
    intensities = np.array([np.mean(img) for img in val_ds.images]) / 255.0

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "depth": depths,
            "coverage": coverages,
            "intensity": intensities,
        }
    )

    # Compute Correlations
    corr_depth = analysis_df["error"].corr(analysis_df["depth"])
    corr_cov = analysis_df["error"].corr(analysis_df["coverage"])
    corr_int = analysis_df["error"].corr(analysis_df["intensity"])

    print(f"Error Correlation with Depth: {corr_depth:.4f}")
    print(f"Error Correlation with Salt Coverage: {corr_cov:.4f}")
    print(f"Error Correlation with Image Intensity: {corr_int:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    if final_metric > 0.827:
        print("\n--- Generating Submission ---")

        # Optimize Threshold
        best_threshold = optimize_threshold(val_preds, val_targets)

        # Predict on Test Set (Phase 2 Model)
        # Note: We already have soft_pseudo_labels from Phase 1, but Phase 2 model is better.
        test_results_final = predict_with_tta(model, test_loader, device)

        # Generate CSV
        generate_submission(
            test_results_final["preds"],
            test_results_final["ids"],
            best_threshold,
            Config.SUBMISSION_PATH,
        )
    else:
        print(f"\nMetric {final_metric:.4f} <= 0.827. Skipping submission generation.")


if __name__ == "__main__":
    main()
