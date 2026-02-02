import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler
from scipy.stats import pointbiserialr

from library.config import Config
from library.utils import (
    seed_everything,
    calculate_macro_f1,
    generate_submission,
    compute_class_priors,
    load_taxonomy_mapping,
)
from library.dataset import get_dataloaders
from library.model import HierarchicalConvNeXt
from library.train import train_one_epoch, validate


def main():
    # ---------------------------------------------------------
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    seed_everything(Config.SEED)
    Config.setup()

    # Override Config for Fast Baseline (2-hour limit)
    # Using 100,000 samples and 1 epoch per stage ensures completion.
    Config.override(
        DEBUG_SAMPLE_SIZE=100000, STAGE1_EPOCHS=1, STAGE2_EPOCHS=1, BATCH_SIZE=128
    )

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")
    print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # ---------------------------------------------------------
    # 2. Model Initialization
    # ---------------------------------------------------------
    print("Initializing Model...")
    model = HierarchicalConvNeXt(pretrained=Config.PRETRAINED)
    model.to(device)

    # Define Loss Functions
    criterion_dict = {
        "species": nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING),
        "family": nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING),
        "order": nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING),
    }

    # ---------------------------------------------------------
    # 3. Stage 1: Representation Learning
    # ---------------------------------------------------------
    print("\n--- Stage 1: Representation Learning ---")
    train_loader_s1, val_loader, test_loader = get_dataloaders(
        stage=1, debug_size=Config.DEBUG_SAMPLE_SIZE
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.STAGE1_LR, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader_s1)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.STAGE1_LR,
        epochs=Config.STAGE1_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )

    scaler = GradScaler()
    best_f1 = -1.0

    for epoch in range(Config.STAGE1_EPOCHS):
        start_time = time.time()
        train_loss = train_one_epoch(
            model, train_loader_s1, optimizer, criterion_dict, device, scaler, scheduler
        )
        val_loss, val_f1 = validate(model, val_loader, criterion_dict, device)
        elapsed = time.time() - start_time

        print(
            f"S1 Epoch {epoch+1} | Time: {elapsed:.1f}s | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.6f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # ---------------------------------------------------------
    # 4. Stage 2: Classifier Re-balancing
    # ---------------------------------------------------------
    print("\n--- Stage 2: Classifier Re-balancing ---")
    # Load best weights from Stage 1
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    model.freeze_backbone()

    # Get Class-Balanced Loader
    train_loader_s2, _, _ = get_dataloaders(
        stage=2, debug_size=Config.DEBUG_SAMPLE_SIZE
    )

    optimizer_s2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.STAGE2_LR,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s2, T_max=Config.STAGE2_EPOCHS
    )

    scaler_s2 = GradScaler()

    for epoch in range(Config.STAGE2_EPOCHS):
        start_time = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader_s2,
            optimizer_s2,
            criterion_dict,
            device,
            scaler_s2,
            scheduler_s2,
        )
        val_loss, val_f1 = validate(model, val_loader, criterion_dict, device)
        elapsed = time.time() - start_time

        print(
            f"S2 Epoch {epoch+1} | Time: {elapsed:.1f}s | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.6f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # ---------------------------------------------------------
    # 5. Final Evaluation & Metric
    # ---------------------------------------------------------
    print("\n--- Final Evaluation ---")
    # Load best model
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Run validation one last time to get exact metrics
    _, final_f1 = validate(model, val_loader, criterion_dict, device)
    print(f"Final Validation Metric: {final_f1}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Failure Analysis ---")
    model.eval()

    # Collect predictions and targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            species_labels = labels[0].to(device)

            outputs = model(images)
            preds = torch.argmax(outputs["species"], dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(species_labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Error (0 for correct, 1 for incorrect)
    errors = (all_preds != all_targets).astype(int)

    # Feature: Class Frequency
    # We map each target class to its frequency in the training set
    class_priors = compute_class_priors(load_cached_data=True)
    target_frequencies = class_priors[all_targets]

    # Correlation between Error and Class Frequency
    # We expect negative correlation (higher frequency -> lower error)
    corr, p_value = pointbiserialr(errors, target_frequencies)

    print(
        f"Correlation between Error and Class Frequency: {corr:.4f} (p-value: {p_value:.4e})"
    )
    if corr < 0:
        print("Insight: Rare classes have higher error rates.")
    else:
        print("Insight: Error rate is not strongly associated with class rarity.")

    # ---------------------------------------------------------
    # 7. Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.48608987524978914

    if final_f1 > THRESHOLD:
        print(
            f"\nMetric ({final_f1}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Use class priors for logit adjustment during inference
        generate_submission(
            model,
            test_loader,
            device,
            class_priors=class_priors,
            output_path=Config.SUBMISSION_PATH,
        )
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nMetric ({final_f1}) <= Threshold ({THRESHOLD}). Skipping submission.")


if __name__ == "__main__":
    main()
