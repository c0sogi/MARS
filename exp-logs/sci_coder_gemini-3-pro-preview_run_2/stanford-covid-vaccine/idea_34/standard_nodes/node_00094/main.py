import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Configuration Overrides
# We import config first to modify settings before other modules load them.
import library.config as config

# Override for Fast Baseline and Path Requirements
config.EPOCHS = 20
config.SUBMISSION_PATH = "./submission/submission.csv"

# Ensure submission directory exists
os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

# Now import the rest of the library modules
from library.utils import set_seed, mcrmse_loss
from library.data import get_dataloaders
from library.model import LFDCN, generate_submission
from library.train import train_one_epoch, validate


def main():
    # Set reproducibility
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=True, batch_size=config.BATCH_SIZE
    )

    # 3. Model Initialization
    model = LFDCN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 4. Training Loop
    print(f"Starting training for {config.EPOCHS} epochs...")
    best_val_score = float("inf")
    early_stop_count = 0

    # Ensure model save path directory exists
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Scheduler
        scheduler.step(val_score)

        # Checkpointing
        if val_score < best_val_score:
            best_val_score = val_score
            early_stop_count = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        else:
            early_stop_count += 1
            if early_stop_count >= config.PATIENCE:
                print("Early stopping triggered.")
                break

    # 5. Final Validation Assessment
    print("\nPerforming Final Validation...")
    # Load best model
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No model file found. Using current model state.")

    # Compute Final Metric
    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.eval()
    val_errors = []
    val_ids = val_loader.dataset.ids

    # We need to compute error per sample to correlate with metadata
    # Iterate through loader again to get per-sample errors
    # Note: validate() computes global metric, here we want sample-wise

    all_sample_errors = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, partner_indices, targets, masks = batch
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            # Inference (Pass 2)
            preds = model(inputs, partner_indices, targets=None)

            # Calculate MCRMSE per sample
            # preds: (B, L, 5), targets: (B, L, 5)
            # Scored indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
            scored_indices = config.SCORED_INDICES

            pred_scored = preds[..., scored_indices]
            target_scored = targets[..., scored_indices]

            # MSE per sample per column: (B, 3)
            # We average over Length (dim 1)
            mse = torch.mean((pred_scored - target_scored) ** 2, dim=1)
            rmse = torch.sqrt(mse)
            # Mean over columns: (B,)
            mcrmse_sample = torch.mean(rmse, dim=1)

            all_sample_errors.extend(mcrmse_sample.cpu().numpy())

    all_sample_errors = np.array(all_sample_errors)

    # Load Metadata for correlation
    val_meta_path = config.VAL_CSV
    if os.path.exists(val_meta_path):
        val_df = pd.read_csv(val_meta_path)

        # Ensure alignment. The dataloader preserves order if shuffle=False (which it is for val)
        # However, let's double check lengths
        if len(val_df) == len(all_sample_errors):
            val_df["error"] = all_sample_errors

            # Calculate correlations
            # Features of interest
            features = ["signal_to_noise", "mean_reactivity", "seq_length"]

            print("Correlation between Error and Metadata Features:")
            for feat in features:
                if feat in val_df.columns:
                    corr = val_df["error"].corr(val_df[feat])
                    print(f"  {feat}: {corr:.4f}")
                else:
                    print(f"  {feat}: Not found in metadata")
        else:
            print(
                f"Warning: Metadata length ({len(val_df)}) does not match prediction length ({len(all_sample_errors)}). Skipping correlation."
            )
    else:
        print("Validation metadata not found. Skipping failure analysis.")

    # 7. Submission Generation
    threshold = 0.5417620723771521
    if final_metric < threshold:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )
        # generate_submission loads the model from MODEL_SAVE_PATH, which we updated with the best model
        generate_submission()
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
