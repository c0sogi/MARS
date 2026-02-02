import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from library
import library.config as config
from library.data import get_loaders, VAL_METADATA_PATH
from library.model import DeepStabilizedBiGRU
from library.utils import MCRMSELoss, compute_mcrmse, SCORED_TARGETS, ALL_TARGETS
from library.train import train_one_epoch, validate, generate_submission

# =============================================================================
# CONFIGURATION OVERRIDES FOR FAST BASELINE
# =============================================================================
# Limit epochs to ensure execution within time limits while allowing convergence
config.EPOCHS = 15
# Ensure we use the full dataset (it is small enough: ~2k samples)
config.DEBUG = False


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set.
    Computes correlations between error magnitude and input features.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # 1. Collect Predictions and Targets
    with torch.no_grad():
        # Access IDs from dataset directly to ensure alignment
        dataset_ids = val_loader.dataset.ids
        start_idx = 0

        for features, pair_indices, pair_masks, targets in val_loader:
            features = features.to(device)
            pair_indices = pair_indices.to(device)
            pair_masks = pair_masks.to(device)
            targets = targets.to(device)

            preds = model(features, pair_indices, pair_masks)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

            batch_size = features.size(0)
            all_ids.extend(dataset_ids[start_idx : start_idx + batch_size])
            start_idx += batch_size

    all_preds = torch.cat(all_preds, dim=0)  # (N, 107, 5)
    all_targets = torch.cat(all_targets, dim=0)  # (N, 107, 5)

    # 2. Calculate Sample-wise Error (MCRMSE on scored columns)
    # Slice to PRED_LEN (68)
    preds_sliced = all_preds[:, : config.PRED_LEN, :]
    targets_sliced = all_targets[:, : config.PRED_LEN, :]

    # Identify scored indices
    scored_indices = [i for i, t in enumerate(ALL_TARGETS) if t in SCORED_TARGETS]

    # Compute squared error: (N, 68, 5) -> (N, 68, 3)
    diff = preds_sliced[:, :, scored_indices] - targets_sliced[:, :, scored_indices]
    mse_per_sample = torch.mean(diff**2, dim=(1, 2))  # Average over seq and targets
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # 3. Load Metadata for Features
    df_val = pd.read_parquet(VAL_METADATA_PATH)
    # Ensure order matches
    df_val = df_val.set_index("id").reindex(all_ids).reset_index()

    # 4. Construct Analysis DataFrame
    analysis_df = pd.DataFrame(
        {
            "error": rmse_per_sample,
            "signal_to_noise": df_val["signal_to_noise"],
            "SN_filter": df_val["SN_filter"],
            "seq_length": df_val["seq_length"],
        }
    )

    # Add sequence features
    analysis_df["pct_A"] = df_val["sequence"].apply(lambda s: s.count("A") / len(s))
    analysis_df["pct_G"] = df_val["sequence"].apply(lambda s: s.count("G") / len(s))
    analysis_df["pct_U"] = df_val["sequence"].apply(lambda s: s.count("U") / len(s))
    analysis_df["pct_C"] = df_val["sequence"].apply(lambda s: s.count("C") / len(s))
    analysis_df["pct_unpaired"] = df_val["structure"].apply(
        lambda s: s.count(".") / len(s)
    )

    # 5. Compute Correlations
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation between Error (MCRMSE) and Features:")
    print(correlations.drop("error"))

    return correlations


def main():
    # Set seed for reproducibility
    config.set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 1. Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 2. Initialize Model
    print("Initializing model...")
    model = DeepStabilizedBiGRU(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        output_dim=config.OUTPUT_DIM,
        dropout=config.DROPOUT,
    ).to(device)

    # 3. Optimization Setup
    criterion = MCRMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # 4. Training Loop
    best_metric = float("inf")
    best_model_path = os.path.join(config.IDEA_DIR, "best_model_runfile.pth")

    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metric = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Metric: {val_metric:.5f}"
        )

        # Save Best Model
        if val_metric < best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Evaluation
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Compute final metric on full validation set
    _, final_val_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 7. Conditional Submission
    THRESHOLD = 0.5884495377540588
    if final_val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_val_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nValidation metric ({final_val_metric}) did not beat threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
