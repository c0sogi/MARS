import os
import torch
import numpy as np
import pandas as pd
import time
from torch.utils.data import DataLoader, Subset

# Import library components
from library.config import Config
from library.preprocess import DataPreprocessor
from library.dataset import MoleculeDataset, collate_molecular_graphs
from library.engine import Trainer
from library.utils import set_seed


def main():
    # ==========================================
    # 1. Configuration & Overrides
    # ==========================================
    # Limit epochs and ensure full data processing (we will subset training data later)
    Config.MAX_EPOCHS = 3
    Config.DEBUG_SAMPLE_SIZE = None
    Config.BATCH_SIZE = 64

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Preprocessing
    # ==========================================
    # Run preprocessing (checks cache first)
    preprocessor = DataPreprocessor()
    preprocessor.process(load_cached_data=True)

    # ==========================================
    # 3. Data Loading
    # ==========================================
    print("Initializing datasets...")
    full_train_dataset = MoleculeDataset(split="train")
    val_dataset = MoleculeDataset(split="val")

    # Optimization: Subset training data for fast baseline execution
    # Using 25,000 molecules (approx 30% of training data)
    subset_indices = range(min(25000, len(full_train_dataset)))
    train_subset = Subset(full_train_dataset, subset_indices)

    print(f"Training on subset: {len(train_subset)} molecules")
    print(f"Validating on full set: {len(val_dataset)} molecules")

    train_loader = DataLoader(
        train_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_molecular_graphs,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_molecular_graphs,
        pin_memory=True,
    )

    # ==========================================
    # 4. Training Loop
    # ==========================================
    trainer = Trainer()

    best_lmae = float("inf")

    print(f"\nStarting training for {Config.MAX_EPOCHS} epochs...")

    for epoch in range(Config.MAX_EPOCHS):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)

        # Validate
        val_metrics = trainer.validate(val_loader)
        val_lmae = val_metrics["LMAE"]

        print(f"Epoch {epoch+1} | Val LMAE: {val_lmae:.6f}")

        # Scheduler Step
        trainer.scheduler.step()

        # Checkpoint
        if val_lmae < best_lmae:
            best_lmae = val_lmae
            print(f"  New best model found! Saving to {Config.BEST_MODEL_PATH}...")
            torch.save(trainer.model.state_dict(), Config.BEST_MODEL_PATH)

    print(f"\nTraining complete. Best Validation LMAE: {best_lmae}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\nRunning Failure Analysis...")

    # Load best model
    trainer.model.load_state_dict(
        torch.load(Config.BEST_MODEL_PATH, map_location=device)
    )
    trainer.model.eval()

    all_errors = []
    all_targets = []
    all_dists = []

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            for key, val in batch.items():
                if torch.is_tensor(val):
                    batch[key] = val.to(device)

            # Predict
            preds_std = trainer.model(batch)
            preds = trainer.standardizer.inverse_transform(
                preds_std, batch["coupling_type"]
            )
            targets = batch["coupling_value"]

            # Calculate Absolute Error
            errors = torch.abs(preds - targets)

            # Calculate Distances
            c_i, c_j = batch["coupling_index"]
            pos = batch["pos"]
            # Vector difference and norm
            dists = (pos[c_i] - pos[c_j]).norm(dim=1)

            all_errors.append(errors.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_dists.append(dists.cpu().numpy())

    # Concatenate
    all_errors = np.concatenate(all_errors)
    all_targets = np.concatenate(all_targets)
    all_dists = np.concatenate(all_dists)

    # Compute Correlations
    corr_target = np.corrcoef(all_errors, np.abs(all_targets))[0, 1]
    corr_dist = np.corrcoef(all_errors, all_dists)[0, 1]

    print(f"Final Validation Metric: {best_lmae}")
    print(f"Correlation (Error vs Target Magnitude): {corr_target:.4f}")
    print(f"Correlation (Error vs Distance): {corr_dist:.4f}")

    # ==========================================
    # 6. Submission
    # ==========================================
    THRESHOLD = -1.2761284112930298

    if best_lmae < THRESHOLD:
        print(
            f"\nMetric ({best_lmae:.4f}) is better than threshold ({THRESHOLD:.4f}). Generating submission..."
        )

        test_dataset = MoleculeDataset(split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_molecular_graphs,
            pin_memory=True,
        )

        submission_ids = []
        submission_preds = []

        with torch.no_grad():
            for batch in test_loader:
                for key, val in batch.items():
                    if torch.is_tensor(val):
                        batch[key] = val.to(device)

                preds_std = trainer.model(batch)
                preds = trainer.standardizer.inverse_transform(
                    preds_std, batch["coupling_type"]
                )

                submission_ids.append(batch["coupling_id"].cpu().numpy())
                submission_preds.append(preds.cpu().numpy())

        # Create DataFrame
        df_sub = pd.DataFrame(
            {
                "id": np.concatenate(submission_ids),
                "scalar_coupling_constant": np.concatenate(submission_preds),
            }
        )

        # Sort by ID to match sample submission format (optional but good practice)
        df_sub.sort_values("id", inplace=True)

        # Save
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({best_lmae:.4f}) did not meet threshold ({THRESHOLD:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
