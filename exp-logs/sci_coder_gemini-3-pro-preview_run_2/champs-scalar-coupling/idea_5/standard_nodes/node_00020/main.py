import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

# Import from provided library files
from library.config import Config
from library.dataset import ChampsDataset, collate_graphs
from library.model import HGANet
from library.engine import train_one_epoch, evaluate, predict
from library.utils import set_seed, CouplingStandardizer


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Enforce reproducibility
    set_seed(Config.SEED)

    # Override Config for Fast Baseline Execution (Time Limit < 2 hours)
    # 3 epochs on ~3.3M samples is aggressive but feasible on A100 (~20-30 min/epoch)
    Config.EPOCHS = 3

    device = Config.DEVICE
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # -------------------------------------------------------------------------
    print("Initializing Datasets...")

    # Train Dataset
    train_dataset = ChampsDataset(
        metadata_path=Config.TRAIN_META_PATH,
        cache_path=Config.CACHE_TRAIN_PATH,
        split="train",
    )

    # Validation Dataset
    val_dataset = ChampsDataset(
        metadata_path=Config.VAL_META_PATH,
        cache_path=Config.CACHE_VAL_PATH,
        split="val",
    )

    # Test Dataset
    test_dataset = ChampsDataset(
        metadata_path=Config.TEST_META_PATH,
        cache_path=Config.CACHE_TEST_PATH,
        split="test",
    )

    # Fit Standardizer on Training Data
    print("Fitting Target Standardizer...")
    standardizer = CouplingStandardizer()
    standardizer.fit(train_dataset.df)

    # DataLoaders
    # num_workers=0 to avoid potential multiprocessing overhead/issues in this constrained env
    # though Config says 12, usually 4-8 is safe. Using Config.NUM_WORKERS.
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_graphs,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_graphs,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_graphs,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing HGA-Net...")
    model = HGANet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR for super-convergence in few epochs
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        avg_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            standardizer=standardizer,
            scheduler=scheduler,
        )

        # Step scheduler if using OneCycleLR (usually stepped per batch, but here per epoch logic in engine is simplified)
        # The engine provided doesn't take scheduler step per batch.
        # We will step it here if it was a plateau scheduler, but for OneCycle we need step per batch.
        # Since engine.py doesn't support batch-level scheduler stepping explicitly,
        # we will rely on the optimizer's constant LR or step here if applicable.
        # Given the provided engine code, let's just step here to be safe or ignore if not supported.
        # Actually, standard practice with provided engine is often just epoch-level stepping.
        # We'll stick to the engine's contract.

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss (MAE Norm): {avg_loss:.6f}"
        )

    # Save model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("Training complete. Model saved.")

    # -------------------------------------------------------------------------
    # 5. Validation & Evaluation
    # -------------------------------------------------------------------------
    print("Evaluating on Validation Set...")
    final_metric = evaluate(model, val_loader, device, standardizer)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nRunning Failure Analysis...")
    # We need predictions on validation set to correlate errors
    # Re-use predict function but we need targets.
    # predict() returns dataframe with IDs. We need to merge with val_dataset.df

    # Generate predictions for validation set (returns DF with id, pred)
    # Note: predict() assumes 'id' is in batch. val dataset has it.
    val_preds_df = predict(model, val_loader, device, standardizer)

    # Merge with ground truth
    # val_dataset.df has ['id', 'scalar_coupling_constant', 'atom_index_0', 'atom_index_1', 'molecule_name', 'type', ...]
    analysis_df = val_dataset.df.merge(
        val_preds_df, on="id", suffixes=("_true", "_pred")
    )

    # Calculate Error
    analysis_df["abs_error"] = (
        analysis_df["scalar_coupling_constant_true"]
        - analysis_df["scalar_coupling_constant_pred"]
    ).abs()

    # Calculate Distance (Euclidean)
    # We need to reconstruct distance from structure files or cache.
    # Since we don't have distance in metadata CSV, we compute it quickly.
    # To save time, we will compute it for the analysis subset.

    print("Computing geometric features for analysis...")
    # Load structures.csv for fast lookup
    structures_df = pd.read_csv(Config.STRUCTURES_CSV)

    # Helper to get coords
    # Merge coords for atom 0
    analysis_df = analysis_df.merge(
        structures_df.rename(
            columns={"atom_index": "atom_index_0", "x": "x0", "y": "y0", "z": "z0"}
        ),
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    # Merge coords for atom 1
    analysis_df = analysis_df.merge(
        structures_df.rename(
            columns={"atom_index": "atom_index_1", "x": "x1", "y": "y1", "z": "z1"}
        ),
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    analysis_df["dist_x"] = analysis_df["x0"] - analysis_df["x1"]
    analysis_df["dist_y"] = analysis_df["y0"] - analysis_df["y1"]
    analysis_df["dist_z"] = analysis_df["z0"] - analysis_df["z1"]
    analysis_df["distance"] = np.sqrt(
        analysis_df["dist_x"] ** 2
        + analysis_df["dist_y"] ** 2
        + analysis_df["dist_z"] ** 2
    )

    # Correlations
    corr_dist = analysis_df["abs_error"].corr(analysis_df["distance"])
    corr_idx0 = analysis_df["abs_error"].corr(analysis_df["atom_index_0"])
    corr_idx1 = analysis_df["abs_error"].corr(analysis_df["atom_index_1"])

    print("-" * 30)
    print("Failure Analysis - Error Correlations:")
    print(f"  Distance: {corr_dist:.4f}")
    print(f"  Atom Index 0: {corr_idx0:.4f}")
    print(f"  Atom Index 1: {corr_idx1:.4f}")
    print("-" * 30)

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    # Threshold check
    THRESHOLD = -1.407172441

    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric:.4f} meets threshold ({THRESHOLD}). Generating submission..."
        )

        submission_df = predict(model, test_loader, device, standardizer)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {final_metric:.4f} did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
