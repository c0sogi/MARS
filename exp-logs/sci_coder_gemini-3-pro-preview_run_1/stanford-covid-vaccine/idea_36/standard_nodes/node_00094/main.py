import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloader
from library.model import UncertaintyAwareBiGRU
from library.loss import UncertaintyAwareMSELoss
from library.engine import train_one_epoch, generate_submission


def get_val_predictions(model, loader, config):
    """
    Runs inference on the validation set and returns predictions and targets.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(config.device)
            loop = batch["loop"].to(config.device)
            dist = batch["pair_dist"].to(config.device)
            target = batch["target"].to(config.device)

            # Forward pass
            pred_val, _ = model(seq, loop, dist)

            # Extract valid positions (0 to 68)
            valid_len = config.pred_len

            # Slice to keep only scored positions: [Batch, 68, 3]
            p = pred_val[:, :valid_len, :].cpu().numpy()
            t = target[:, :valid_len, :].cpu().numpy()

            all_preds.append(p)
            all_targets.append(t)

    # Concatenate all batches: [N_samples, 68, 3]
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return all_preds, all_targets


def compute_mcrmse(preds, targets):
    """
    Computes Mean Columnwise Root Mean Squared Error.
    preds, targets: Arrays of shape [N, 3] (flattened spatial dim)
    """
    rmse_list = []
    for i in range(3):
        # MSE for this column
        mse = np.mean((preds[:, i] - targets[:, i]) ** 2)
        rmse_list.append(np.sqrt(mse))
    return np.mean(rmse_list)


def main():
    # 1. Setup
    seed_everything(42)
    config = Config()

    # Ensure working directories exist
    os.makedirs(config.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)

    print(f"Device: {config.device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Load cached data if available
    train_loader = get_dataloader(
        mode="train", config=config, shuffle=True, load_cached_data=True
    )
    val_loader = get_dataloader(
        mode="val", config=config, shuffle=False, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = UncertaintyAwareBiGRU(config).to(config.device)

    # 4. Optimization
    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = UncertaintyAwareMSELoss(lambda_uncertainty=1.0)

    # 5. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(config.cache_dir, "best_model.pth")

    print("Starting training...")
    for epoch in range(config.epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, config)

        # Validation
        val_preds, val_targets = get_val_predictions(model, val_loader, config)

        # Flatten predictions for metric calculation: [N_samples * 68, 3]
        flat_preds = val_preds.reshape(-1, 3)
        flat_targets = val_targets.reshape(-1, 3)

        val_score = compute_mcrmse(flat_preds, flat_targets)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Evaluation
    print(f"Final Validation Metric: {best_score}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=config.device))

    # Get predictions on validation set
    val_preds, val_targets = get_val_predictions(model, val_loader, config)

    # Compute error per sample (Mean RMSE across the 3 targets)
    # val_preds shape: [N_samples, 68, 3]
    sample_errors = []
    for i in range(len(val_preds)):
        p = val_preds[i]
        t = val_targets[i]
        # RMSE per column for this sample
        col_rmses = np.sqrt(np.mean((p - t) ** 2, axis=0))
        # Mean of the 3 column RMSEs
        sample_errors.append(np.mean(col_rmses))

    sample_errors = np.array(sample_errors)

    # Load metadata for correlation analysis
    # Note: val_loader was created with shuffle=False, so order matches the parquet file
    val_df = pd.read_parquet(config.val_parquet)

    # Extract features
    features = {}

    if "signal_to_noise" in val_df.columns:
        features["signal_to_noise"] = val_df["signal_to_noise"].values

    if "SN_filter" in val_df.columns:
        features["SN_filter"] = val_df["SN_filter"].values.astype(float)

    features["seq_length"] = val_df["seq_length"].values

    # Derived feature: GC Content
    features["GC_content"] = (
        val_df["sequence"]
        .apply(lambda x: (x.count("G") + x.count("C")) / len(x))
        .values
    )

    print("Correlation between Error (MCRMSE per sample) and Features:")
    for name, vals in features.items():
        try:
            # Ensure values are float
            vals = vals.astype(float)
            if len(vals) == len(sample_errors):
                corr, _ = pearsonr(sample_errors, vals)
                print(f"  {name}: {corr:.4f}")
            else:
                print(
                    f"  {name}: Length mismatch ({len(vals)} vs {len(sample_errors)})"
                )
        except Exception as e:
            print(f"  {name}: Correlation calculation failed ({e})")

    # 8. Submission
    THRESHOLD = 0.6199890971183777
    if best_score < THRESHOLD:
        print(
            f"\nValidation score ({best_score:.6f}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, config)
    else:
        print(
            f"\nValidation score ({best_score:.6f}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
