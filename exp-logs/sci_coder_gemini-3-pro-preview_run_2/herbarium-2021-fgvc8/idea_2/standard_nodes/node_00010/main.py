import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Import from library
from library.config import Config
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    calculate_f1,
)
from library.taxonomy import TaxonomyManager
from library.dataset import HerbariumDataset, get_transforms, CutMixCollator
from library.model import HierarchicalModel
from library.loss import HierarchicalLoss
from library.engine import train_one_epoch, evaluate


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Update Config for Scaled Training
    Config.BATCH_SIZE = 1024  # Maximize batch size for throughput
    Config.EPOCHS = 3  # Sufficient for convergence with large data

    # Setup directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Taxonomy & Data Preparation
    # -------------------------------------------------------------------------
    print("Initializing Taxonomy Manager...")
    tm = TaxonomyManager()
    tm.process_taxonomy(load_cached_data=True)
    num_families, num_orders = tm.get_counts()
    print(f"Taxonomy: {num_families} Families, {num_orders} Orders")

    print("Preparing Datasets...")
    # Training Dataset
    train_dataset = HerbariumDataset(
        mode="train", transform=get_transforms("train", Config.IMAGE_SIZE)
    )

    # Subsample training data for scaled training (1.5M samples)
    # The dataframe is already shuffled in the metadata generation step
    FAST_TRAIN_SIZE = 1500000
    if len(train_dataset.df) > FAST_TRAIN_SIZE:
        print(
            f"Subsampling training set from {len(train_dataset.df)} to {FAST_TRAIN_SIZE} for scaled training."
        )
        train_dataset.df = train_dataset.df.iloc[:FAST_TRAIN_SIZE].reset_index(
            drop=True
        )

    # Validation Dataset (Use full set for accurate metric)
    val_dataset = HerbariumDataset(
        mode="val", transform=get_transforms("val", Config.IMAGE_SIZE)
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=CutMixCollator(alpha=1.0, p=0.5),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing Model...")
    model = HierarchicalModel(
        backbone_name=Config.BACKBONE,
        pretrained=True,
        num_classes=Config.NUM_CLASSES,
        num_families=num_families,
        num_orders=num_orders,
        embedding_dim=Config.EMBEDDING_DIM,
        dropout=Config.DROPOUT,
    )
    model = model.to(device)

    # -------------------------------------------------------------------------
    # 4. Training Setup
    # -------------------------------------------------------------------------
    criterion = HierarchicalLoss().to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # One Cycle Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=100.0,
    )

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    best_f1 = 0.0

    print("Starting Training...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )

        # Validate
        val_f1, val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.6f}"
        )

        # Save Best Model
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_f1": best_f1,
                    "optimizer": optimizer.state_dict(),
                },
                filename=Config.MODEL_CHECKPOINT_PATH,
            )

    # -------------------------------------------------------------------------
    # 6. Final Evaluation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Final Evaluation & Failure Analysis ---")

    # Load best model
    print("Loading best model for analysis...")
    checkpoint = load_checkpoint(
        model, filename=Config.MODEL_CHECKPOINT_PATH, device=device
    )
    if checkpoint:
        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} with F1: {checkpoint.get('best_f1', 0):.6f}"
        )

    # Re-run evaluation to get predictions for analysis
    model.eval()
    all_preds = []
    all_targets = []

    # We need a custom loop here to gather data for failure analysis efficiently
    # reusing the logic from evaluate but keeping the data
    with torch.no_grad():
        for images, species, families, orders in val_loader:
            images = images.to(device)
            # Forward pass (inference mode)
            outputs = model(images, species_label=None)
            species_logits = outputs[0]
            preds = torch.argmax(species_logits, dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(species)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    final_f1 = calculate_f1(all_targets, all_preds)
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis: Correlation between Error and Class Frequency
    # 1. Calculate correctness
    correctness = (all_preds == all_targets).astype(int)

    # 2. Calculate Class Counts in Training Set (using the full training csv for accuracy of distribution)
    # We use the full metadata df to get the 'true' distribution, not just the subsample
    full_train_df = pd.read_csv(Config.TRAIN_CSV)
    class_counts = full_train_df["category_id"].value_counts().to_dict()

    # 3. Map targets to their training frequency
    target_counts = np.array([class_counts.get(t, 0) for t in all_targets])

    # 4. Calculate Correlation
    if len(np.unique(correctness)) > 1:
        correlation = np.corrcoef(correctness, target_counts)[0, 1]
    else:
        correlation = 0.0

    print(
        f"Failure Analysis - Correlation (Correctness vs Class Frequency): {correlation:.6f}"
    )

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.48608987524978914

    if final_f1 > THRESHOLD:
        print(
            f"\nValidation metric ({final_f1}) > threshold ({THRESHOLD}). Generating submission..."
        )

        # Prepare Test Loader
        test_dataset = HerbariumDataset(
            mode="test", transform=get_transforms("val", Config.IMAGE_SIZE)
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        submission_ids = []
        submission_preds = []

        model.eval()
        with torch.no_grad():
            for images, image_ids in test_loader:
                images = images.to(device)

                # Inference
                outputs = model(images, species_label=None)
                species_logits = outputs[0]
                preds = torch.argmax(species_logits, dim=1).cpu().numpy()

                submission_ids.extend(image_ids.numpy())
                submission_preds.extend(preds)

        # Create DataFrame
        df_sub = pd.DataFrame({"Id": submission_ids, "Predicted": submission_preds})

        # Save
        df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"\nValidation metric ({final_f1}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
