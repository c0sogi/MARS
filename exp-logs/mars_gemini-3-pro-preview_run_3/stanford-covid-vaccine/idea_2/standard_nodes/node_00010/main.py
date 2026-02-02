import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed, mcrmse_loss, format_submission
from library.data import get_dataloaders
from library.model import HybridGNN
from library.train import Trainer


def main():
    # 1. Setup and Configuration
    # Override Config for a fast baseline run
    Config.NUM_EPOCHS = 15  # Reduced for speed
    Config.BATCH_SIZE = 32

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # 2. Data Loading
    # We use load_cached_data=True to leverage pre-processed tensors if available
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    model = HybridGNN().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # Fit the model
    trainer.fit(num_epochs=Config.NUM_EPOCHS, patience=Config.EARLY_STOPPING_PATIENCE)

    # 5. Evaluation & Metrics
    # Load the best model state
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Run inference on Validation set
    val_preds_list = []
    val_targets_list = []
    val_ids_list = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch)
            targets = batch.y

            # Reshape targets if necessary (PyG batching flattens dim 0)
            batch_size = preds.size(0)
            seq_len = Config.SEQ_LENGTH
            if targets.size(0) == batch_size * seq_len:
                targets = targets.view(batch_size, seq_len, -1)

            val_preds_list.append(preds.cpu())
            val_targets_list.append(targets.cpu())
            val_ids_list.extend(batch.id)

    val_preds = torch.cat(val_preds_list, dim=0)
    val_targets = torch.cat(val_targets_list, dim=0)

    # Compute Final Metric
    final_metric = mcrmse_loss(val_preds, val_targets).item()
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Load validation metadata to get features like Signal-to-Noise
    val_meta_df = pd.read_parquet(Config.VAL_DATA_PATH)

    # Calculate error per sample (MCRMSE per sample)
    # val_preds: (N, 107, 5), val_targets: (N, 107, 5)
    # We only score the first 68 positions
    scored_len = Config.SEQ_SCORED

    # Error tensor: (N, 68, 5)
    diff = val_preds[:, :scored_len, :] - val_targets[:, :scored_len, :]
    mse_per_sample = torch.mean(diff**2, dim=(1, 2))  # Average over seq and targets
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame({"id": val_ids_list, "error": rmse_per_sample})

    # Merge with metadata
    analysis_df = analysis_df.merge(val_meta_df, on="id", how="left")

    # Feature Engineering for correlation
    # 1. Signal to Noise
    # 2. Sequence Length (Constant 107, so ignore)
    # 3. GC Content
    analysis_df["gc_content"] = analysis_df["sequence"].apply(
        lambda s: (s.count("G") + s.count("C")) / len(s)
    )
    analysis_df["pct_paired"] = analysis_df["structure"].apply(
        lambda s: (s.count("(") + s.count(")")) / len(s)
    )

    features_to_check = ["signal_to_noise", "gc_content", "pct_paired"]

    print("Correlation between Error and Input Features:")
    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Drop NaNs if any
            valid_data = analysis_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["error"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not enough data")
        else:
            print(f"  {feat}: Column not found")

    # 7. Submission Generation
    # Threshold check
    THRESHOLD = 0.7421537041664124

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds_list = []
        test_ids_list = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                preds = model(batch)

                test_preds_list.append(preds.cpu())
                test_ids_list.extend(batch.id)

        test_preds = torch.cat(test_preds_list, dim=0)

        # Format and Save
        submission_df = format_submission(
            ids=test_ids_list, preds=test_preds, save_path=Config.SUBMISSION_PATH
        )
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {submission_df.shape}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
