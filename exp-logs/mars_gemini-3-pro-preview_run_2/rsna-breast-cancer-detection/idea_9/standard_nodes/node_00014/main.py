import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Subset

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, probabilistic_f1
from library.data import get_dataloaders
from library.model import SHR_MTN
from library.train import train_one_epoch, validate_one_epoch, generate_submission


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # Override Config for Fast Baseline
    Config.NUM_EPOCHS = 2
    TRAIN_SUBSET_SIZE = 5000  # Limit training data for speed

    # 2. Data Loading
    # Load full datasets to ensure validation is done on the entire hold-out set
    print("Loading DataLoaders...")
    train_loader_full, val_loader, test_loader = get_dataloaders(
        debug=False, load_cached_data=True
    )

    # Create a subset of training data for the fast baseline
    # We use the first N samples. Since the loader shuffles, we subset the dataset directly.
    train_dataset = train_loader_full.dataset
    subset_indices = range(min(len(train_dataset), TRAIN_SUBSET_SIZE))
    train_subset = Subset(train_dataset, subset_indices)

    train_loader_fast = DataLoader(
        train_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )
    print(f"Training on subset of {len(train_subset)} samples.")
    print(f"Validating on full set of {len(val_loader.dataset)} samples.")

    # 3. Model Initialization
    num_aux_features = len(train_dataset.feature_cols)
    model = SHR_MTN(num_aux_features=num_aux_features)
    model.to(device)

    # 4. Optimization & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scaler = amp.GradScaler(enabled=True)

    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criteria = {
        "cancer": nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        "birads": nn.MSELoss(),
        "density": nn.CrossEntropyLoss(),
    }

    # 5. Training Loop
    best_pf1 = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("\n=== Starting Training ===")
    for epoch in range(Config.NUM_EPOCHS):
        # Train on subset
        t_loss, t_cancer, t_pf1 = train_one_epoch(
            model, train_loader_fast, optimizer, criteria, scaler, device
        )

        # Validate on full set
        v_loss, v_cancer, v_pf1 = validate_one_epoch(
            model, val_loader, criteria, device
        )

        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"  Train: Loss={t_loss:.4f}, CancerLoss={t_cancer:.4f}, pF1={t_pf1:.4f}")
        print(f"  Val:   Loss={v_loss:.4f}, CancerLoss={v_cancer:.4f}, pF1={v_pf1:.4f}")

        if v_pf1 > best_pf1:
            print(f"  [Improvement] Saving model (pF1: {best_pf1:.4f} -> {v_pf1:.4f})")
            best_pf1 = v_pf1
            torch.save(model.state_dict(), best_model_path)
        else:
            print(f"  [No Improvement] Best pF1: {best_pf1:.4f}")

    # 6. Final Validation & Metric Calculation
    print("\n=== Final Validation Analysis ===")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()
    val_preds = []
    val_targets = []

    # Run inference manually to get arrays for analysis
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            aux = batch["aux_features"].to(device)
            targets = batch["targets"]["cancer"].numpy()

            # Inference
            with amp.autocast(enabled=True):
                outputs = model(images, aux)

            # Get probabilities
            probs = torch.sigmoid(outputs["cancer"].float()).squeeze(1).cpu().numpy()

            val_preds.extend(probs)
            val_targets.extend(targets)

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    final_metric = probabilistic_f1(val_targets, val_preds)
    # Required Output Format
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(val_preds - val_targets)

    # Get metadata for correlation
    val_df = val_loader.dataset.df

    # Ensure alignment
    if len(val_df) != len(errors):
        print(
            f"Warning: DataFrame length ({len(val_df)}) != Predictions length ({len(errors)}). Truncating to minimum."
        )
        min_len = min(len(val_df), len(errors))
        val_df = val_df.iloc[:min_len]
        errors = errors[:min_len]

    # Features to analyze
    features_to_check = ["age", "density_label", "implant_enc", "lat_enc"]
    # Add view features dynamically
    features_to_check.extend([c for c in val_df.columns if c.startswith("view_")])

    for feature in features_to_check:
        if feature in val_df.columns:
            vals = val_df[feature].values
            # Check if numeric
            if np.issubdtype(vals.dtype, np.number):
                # Handle potential NaNs or constant values
                if np.std(vals) == 0 or np.std(errors) == 0:
                    corr = 0.0
                else:
                    corr = np.corrcoef(vals, errors)[0, 1]

                print(f"Correlation (Error vs {feature}): {corr:.4f}")

    # 8. Submission Generation
    THRESHOLD = 0.044888656586408615

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
