import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import prepare_data
from library.model import DSPIN
from library.engine import fit, evaluate, generate_submission, predict
from library.feature_engineering import load_and_process_data


def get_feature_names():
    """
    Helper to retrieve feature names by processing a tiny subset of data.
    This ensures we match the columns used in the dataset.
    """
    # Load a debug version to get columns quickly
    df = load_and_process_data("train", load_cached_data=False, debug=True)
    exclude_cols = {
        Config.ID_COL,
        Config.BREATH_ID_COL,
        Config.TARGET_COL,
        "source_file",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Preparation
    print("Preparing data...")
    # load_cached_data=True allows using pre-computed files if available in ./working
    train_dataset, val_dataset, test_dataset, scaler = prepare_data(
        load_cached_data=True, debug=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    # Determine input dimension from dataset
    input_dim = train_dataset.X.shape[-1]
    print(f"Input dimension: {input_dim}")

    model = DSPIN(input_dim=input_dim).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 4. Training
    print("Starting training...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
    )

    # 5. Evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    from library.engine import WeightedL1Loss

    criterion = WeightedL1Loss()

    val_loss, val_mae = evaluate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_mae}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Get predictions and targets
    # We need to manually get preds to correlate with features
    model.eval()
    val_preds = []
    val_targets = []
    val_u_out = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u = batch["u_out"].to(device)

            p = model(x).squeeze(-1)

            val_preds.append(p.cpu().numpy())
            val_targets.append(y.cpu().numpy())
            val_u_out.append(u.cpu().numpy())

    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()
    val_u_out = np.concatenate(val_u_out).flatten()

    # Calculate Error (Absolute)
    errors = np.abs(val_preds - val_targets)

    # Filter for Inspiratory phase only (as per metric)
    insp_mask = val_u_out == 0
    insp_errors = errors[insp_mask]

    # Get Feature Names
    feature_names = get_feature_names()

    # Get Validation Features (Flattened)
    # val_dataset.X is (N_breaths, 80, N_features)
    X_val_flat = val_dataset.X.numpy().reshape(-1, input_dim)
    X_val_insp = X_val_flat[insp_mask]

    # Calculate Correlations
    print("Correlation between Absolute Error and Features (Inspiratory Phase):")
    correlations = {}
    for i, feat_name in enumerate(feature_names):
        if i < X_val_insp.shape[1]:
            feat_values = X_val_insp[:, i]
            # Handle potential constant features to avoid NaN
            if np.std(feat_values) > 1e-9:
                corr = np.corrcoef(insp_errors, feat_values)[0, 1]
                correlations[feat_name] = corr
            else:
                correlations[feat_name] = 0.0

    # Sort and Print Top Correlations
    sorted_corr = sorted(
        correlations.items(), key=lambda item: abs(item[1]), reverse=True
    )
    for name, corr in sorted_corr[:10]:
        print(f"{name}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.20567339658737183

    if val_mae < THRESHOLD:
        print(
            f"\nValidation Metric ({val_mae}) < Threshold ({THRESHOLD}). Generating submission..."
        )

        # Ensure output directory exists
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)

        # Override path in Config
        Config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nValidation Metric ({val_mae}) >= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
