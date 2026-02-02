import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.taxonomy import get_taxonomy_mappings
from library.dataset import HerbariumDataset
from library.model import HierarchicalEfficientNet
from library.loss import HierarchicalLoss
from library.engine import train_one_epoch, validate, predict


def run_failure_analysis(model, dataloader, device, dataset_df):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error and metadata features.
    """
    model.eval()
    results = []

    print("\nRunning Failure Analysis on Validation Set...")

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            # Targets
            targets = {
                "species_id": batch["species_id"].to(device),
            }
            img_ids = batch["image_id"].numpy()

            outputs = model(images)

            # Species prediction
            species_logits = outputs["species"]
            preds = torch.argmax(species_logits, dim=1).cpu().numpy()
            labels = targets["species_id"].cpu().numpy()

            for img_id, pred, label in zip(img_ids, preds, labels):
                results.append(
                    {
                        "image_id": img_id,
                        "predicted": pred,
                        "target": label,
                        "error": int(pred != label),
                    }
                )

    results_df = pd.DataFrame(results)

    # Merge with metadata to get features
    # dataset_df has image_id, region_id, family_id, genus_id
    # We ensure we only merge necessary columns to avoid duplicates if any
    meta_cols = ["image_id", "region_id", "family_id", "genus_id"]
    meta_subset = dataset_df[meta_cols]

    analysis_df = results_df.merge(meta_subset, on="image_id", how="left")

    # Calculate correlations
    print("\n==== Failure Analysis Correlations ====")
    features = ["region_id", "family_id", "genus_id"]
    for feat in features:
        if feat in analysis_df.columns:
            # Calculate correlation between binary error and the feature ID
            corr = analysis_df["error"].corr(analysis_df[feat])
            print(f"Correlation between Error and {feat}: {corr}")
        else:
            print(f"Feature {feat} not found in dataframe.")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Override Config for Fast Baseline Execution
    # We limit the training set size and epochs to ensure we finish within the time limit
    MAX_TRAIN_SAMPLES = 80000
    PHASE1_EPOCHS = 1
    PHASE2_EPOCHS = 1

    # 2. Load Taxonomy
    taxonomy_df = get_taxonomy_mappings(load_cached_data=True)
    n_species = len(taxonomy_df)
    n_genera = taxonomy_df["genus_id"].nunique()
    n_families = taxonomy_df["family_id"].nunique()

    print(
        f"Model Configuration: {n_species} Species, {n_genera} Genera, {n_families} Families"
    )

    # 3. Initialize Model
    model = HierarchicalEfficientNet(n_families, n_genera, n_species, pretrained=True)
    model.to(device)

    # Define Loss
    loss_fn = HierarchicalLoss()

    # =========================================================================
    # PHASE 1: Coarse Resolution (224x224)
    # =========================================================================
    print("\n==== Starting Phase 1: 224x224 ====")

    # Dataset & Loader
    train_dataset_p1 = HerbariumDataset(split="train", image_size=224)

    # Subsample training data for speed
    if len(train_dataset_p1.df) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(train_dataset_p1.df)} to {MAX_TRAIN_SAMPLES}..."
        )
        train_dataset_p1.df = train_dataset_p1.df.sample(
            n=MAX_TRAIN_SAMPLES, random_state=Config.SEED
        ).reset_index(drop=True)
        # Update internal arrays after subsampling
        train_dataset_p1.species_ids = train_dataset_p1.df["category_id"].values
        train_dataset_p1.genus_ids = train_dataset_p1.df["genus_id"].values
        train_dataset_p1.family_ids = train_dataset_p1.df["family_id"].values
        train_dataset_p1.file_paths = train_dataset_p1.df["file_path"].values
        train_dataset_p1.image_ids = train_dataset_p1.df["image_id"].values

    train_loader_p1 = DataLoader(
        train_dataset_p1,
        batch_size=Config.PHASE1_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Use full validation set for monitoring
    val_dataset_p1 = HerbariumDataset(split="val", image_size=224)
    val_loader_p1 = DataLoader(
        val_dataset_p1,
        batch_size=Config.PHASE1_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.PHASE1_LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Train Loop Phase 1
    best_f1_p1 = -1.0

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader_p1, optimizer, device, loss_fn)
        val_loss, val_f1 = validate(model, val_loader_p1, device, loss_fn)
        print(
            f"Phase 1 Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1_p1:
            best_f1_p1 = val_f1
            torch.save(model.state_dict(), Config.CHECKPOINT_PHASE1)

    # Load best phase 1 model
    if os.path.exists(Config.CHECKPOINT_PHASE1):
        model.load_state_dict(torch.load(Config.CHECKPOINT_PHASE1, map_location=device))
        print("Loaded best Phase 1 checkpoint.")

    # =========================================================================
    # PHASE 2: Fine Resolution (300x300)
    # =========================================================================
    print("\n==== Starting Phase 2: 300x300 ====")

    train_dataset_p2 = HerbariumDataset(split="train", image_size=300)
    # Apply same subsampling
    if len(train_dataset_p2.df) > MAX_TRAIN_SAMPLES:
        train_dataset_p2.df = train_dataset_p2.df.sample(
            n=MAX_TRAIN_SAMPLES, random_state=Config.SEED
        ).reset_index(drop=True)
        train_dataset_p2.species_ids = train_dataset_p2.df["category_id"].values
        train_dataset_p2.genus_ids = train_dataset_p2.df["genus_id"].values
        train_dataset_p2.family_ids = train_dataset_p2.df["family_id"].values
        train_dataset_p2.file_paths = train_dataset_p2.df["file_path"].values
        train_dataset_p2.image_ids = train_dataset_p2.df["image_id"].values

    train_loader_p2 = DataLoader(
        train_dataset_p2,
        batch_size=Config.PHASE2_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_dataset_p2 = HerbariumDataset(split="val", image_size=300)
    val_loader_p2 = DataLoader(
        val_dataset_p2,
        batch_size=Config.PHASE2_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Lower LR for fine-tuning
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.PHASE2_LR, weight_decay=Config.WEIGHT_DECAY
    )

    best_f1_final = -1.0

    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader_p2, optimizer, device, loss_fn)
        val_loss, val_f1 = validate(model, val_loader_p2, device, loss_fn)
        print(
            f"Phase 2 Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1_final:
            best_f1_final = val_f1
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # Load best model for final evaluation
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best Phase 2 checkpoint.")

    # =========================================================================
    # Final Evaluation & Failure Analysis
    # =========================================================================
    print("\n==== Final Evaluation ====")
    # Validate on FULL validation set
    final_loss, final_f1 = validate(model, val_loader_p2, device, loss_fn)
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis
    run_failure_analysis(model, val_loader_p2, device, val_dataset_p2.df)

    # =========================================================================
    # Submission
    # =========================================================================
    THRESHOLD = 0.43008749389564027

    if final_f1 > THRESHOLD:
        print(f"\nMetric {final_f1} > {THRESHOLD}. Generating submission...")
        test_dataset = HerbariumDataset(split="test", image_size=300)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.PHASE2_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        predict(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(f"\nMetric {final_f1} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
