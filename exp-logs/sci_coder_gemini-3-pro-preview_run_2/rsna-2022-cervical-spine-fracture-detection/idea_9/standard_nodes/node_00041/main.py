import os
import sys
import importlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Explicitly reload libraries to ensure Config changes are picked up (Cite debug_lesson_4)
import library.config
import library.engine

importlib.reload(library.config)
importlib.reload(library.engine)

# Import from provided library files
from library.config import Config, seed_everything
from library.dataset import CervicalSpineDataset, get_slice_cache, get_bbox_cache
from library.model import CervicalFractureNet
from library.engine import fit, evaluate, generate_submission
from library.utils import load_checkpoint, calculate_weighted_log_loss


def perform_failure_analysis(model, val_loader, val_metadata, slice_cache, device):
    """
    Analyzes model performance on the validation set to identify error patterns.
    Correlates study-level loss with input features (e.g., number of slices).
    """
    model.eval()

    study_losses = []
    slice_counts = []

    # We need to map batch predictions back to studies to calculate per-study loss
    # Since the loader is sequential (shuffle=False), we can iterate metadata
    meta_idx = 0

    # Weights for the metric: C1-C7 (1.0), patient_overall (7.0)
    class_weights = torch.tensor([1.0] * 7 + [7.0], device=device)

    print("\n=== Failure Analysis ===")

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            y_true = targets["study_labels"].to(device)  # (B, 8)

            outputs = model(images)
            y_pred = torch.sigmoid(outputs["study_logits"])  # (B, 8)

            # Calculate weighted log loss per study (row-wise mean of weighted class losses)
            # Loss = - w * [y log p + (1-y) log (1-p)]
            epsilon = 1e-7
            y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

            term1 = y_true * torch.log(y_pred)
            term2 = (1 - y_true) * torch.log(1 - y_pred)
            loss_per_class = -(term1 + term2) * class_weights

            # Average over classes to get a scalar loss per study
            # Note: The competition metric averages over all rows.
            # Here we compute the contribution of each study to that average.
            batch_losses = loss_per_class.mean(dim=1).cpu().numpy()

            batch_size = images.size(0)

            for b in range(batch_size):
                uid = val_metadata.iloc[meta_idx]["StudyInstanceUID"]

                # Get metadata feature: Slice Count
                # slice_cache is {uid: [slice_nums]}
                num_slices = len(slice_cache.get(uid, []))

                study_losses.append(batch_losses[b])
                slice_counts.append(num_slices)

                meta_idx += 1

    # Calculate Correlation
    if len(study_losses) > 1:
        corr, _ = pearsonr(study_losses, slice_counts)
        print(f"Correlation between Error Magnitude and Slice Count: {corr:.4f}")
    else:
        print("Insufficient validation samples for correlation analysis.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for Fast Baseline if needed
    # We stick to Config defaults (10 epochs) as dataset is small (161 samples)

    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading metadata...")
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Initialize Caches
    # We combine train/val/test uids to build/load cache once if possible,
    # but get_slice_cache logic handles lists.
    print("Initializing caches...")
    slice_cache = get_slice_cache(
        pd.concat([train_meta, val_meta]), load_cached_data=True
    )
    bbox_cache = get_bbox_cache(Config.BOUNDING_BOX_PATH, load_cached_data=True)

    # Create Datasets
    train_ds = CervicalSpineDataset(
        train_meta, slice_cache, bbox_cache, is_train=True, seq_len=Config.SEQ_LEN
    )

    val_ds = CervicalSpineDataset(
        val_meta,
        slice_cache,
        bbox_cache,
        is_train=True,  # True to return targets
        seq_len=Config.SEQ_LEN,
    )

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = CervicalFractureNet(pretrained=True).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_steps = len(train_loader) * Config.EPOCHS // Config.ACCUMULATION_STEPS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=Config.MIN_LR
    )

    # 4. Training
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_dir=Config.OUTPUT_DIR,
    )

    # 5. Validation & Metrics
    print("Loading best model for final validation...")
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    load_checkpoint(best_model_path, model)

    val_score = evaluate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, val_meta, slice_cache, device)

    # 7. Conditional Submission
    THRESHOLD = 0.15364714496434773

    if val_score < THRESHOLD:
        print(
            f"Validation score ({val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, device)
    else:
        print(
            f"Validation score ({val_score}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
