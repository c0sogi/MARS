import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.trainer import Trainer
from library.dataset import RNAGraphDataset
from library.model import RNAGNN


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("Setting up configuration for fast baseline...")
    # Override Config for a faster run
    Config.EPOCHS = 20
    Config.PATIENCE = 5

    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    print("Initializing Trainer...")
    # load_cached_data=True allows using pre-processed .pt files if available
    trainer = Trainer(load_cached_data=True)

    print("Starting Training...")
    trainer.fit()

    # -------------------------------------------------------------------------
    # 3. Validation Inference & Metric Calculation
    # -------------------------------------------------------------------------
    print("Loading best model for validation analysis...")
    device = torch.device(Config.DEVICE)
    model = RNAGNN().to(device)

    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found at {Config.BEST_MODEL_PATH}")

    model.load_state_dict(
        torch.load(Config.BEST_MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    print("Loading Validation Dataset...")
    val_dataset = RNAGraphDataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Containers for global metric calculation
    all_preds = []
    all_targets = []
    all_masks = []
    all_ids = []

    print("Running Validation Inference...")
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch)

            # Unbatching: PyG batches stack nodes vertically.
            # We reshape to (Batch_Size, Seq_Len, Num_Targets)
            # We assume constant sequence length of 107 as defined in Config
            num_graphs = len(batch.id)

            preds = preds.view(num_graphs, Config.SEQ_LEN, Config.NUM_TARGETS)
            targets = batch.y.view(num_graphs, Config.SEQ_LEN, Config.NUM_TARGETS)
            mask = batch.mask.view(num_graphs, Config.SEQ_LEN)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(mask.cpu())
            all_ids.extend(batch.id)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)  # (N_samples, 107, 5)
    all_targets = torch.cat(all_targets, dim=0)  # (N_samples, 107, 5)
    all_masks = torch.cat(all_masks, dim=0)  # (N_samples, 107)

    # Calculate MCRMSE (Mean Columnwise Root Mean Squared Error)
    # Metric: Average of RMSEs for each of the scored target columns
    rmse_list = []
    for col in Config.SCORED_INDICES:
        # Flatten predictions and targets for this column
        p_col = all_preds[:, :, col]
        t_col = all_targets[:, :, col]

        # Apply mask: select only scored positions
        # Note: mask is boolean (N_samples, 107)
        p_masked = p_col[all_masks]
        t_masked = t_col[all_masks]

        # Calculate MSE -> RMSE
        mse = torch.mean((p_masked - t_masked) ** 2)
        rmse = torch.sqrt(mse)
        rmse_list.append(rmse.item())

    final_metric = np.mean(rmse_list)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate error per sample for correlation analysis
    # We use RMSE per sample over masked positions
    sample_errors = []

    # Calculate squared errors: (N, 107, 5)
    squared_diffs = (all_preds - all_targets) ** 2

    # Select only scored columns for analysis
    squared_diffs = squared_diffs[:, :, Config.SCORED_INDICES]

    # Mean over the scored targets: (N, 107)
    mean_sq_diff_per_pos = torch.mean(squared_diffs, dim=2)

    for i in range(len(all_ids)):
        mask_i = all_masks[i]
        if mask_i.sum() > 0:
            # Mean over scored positions
            mse_i = torch.mean(mean_sq_diff_per_pos[i][mask_i])
            rmse_i = torch.sqrt(mse_i).item()
            sample_errors.append(rmse_i)
        else:
            sample_errors.append(0.0)

    # Create DataFrame for analysis
    df_errors = pd.DataFrame({"id": all_ids, "error": sample_errors})

    # Load metadata to get features
    if os.path.exists(Config.VAL_METADATA_PATH):
        df_val_meta = pd.read_parquet(Config.VAL_METADATA_PATH)

        # Merge errors with metadata
        df_analysis = pd.merge(df_errors, df_val_meta, on="id", how="inner")

        # Feature Engineering for correlation
        # 1. Signal to Noise (already in metadata)
        # 2. SN_filter (already in metadata)
        # 3. Sequence Composition (e.g., count of Adenine)
        df_analysis["len_A"] = df_analysis["sequence"].apply(lambda x: x.count("A"))

        features_to_analyze = ["signal_to_noise", "SN_filter", "len_A"]

        print("Correlation between Error Magnitude and Input Features:")
        for feat in features_to_analyze:
            if feat in df_analysis.columns:
                # Drop NaNs to ensure valid correlation calculation
                valid_data = df_analysis[[feat, "error"]].dropna()
                if len(valid_data) > 1:
                    corr, _ = pearsonr(valid_data[feat], valid_data["error"])
                    print(f"  {feat}: {corr:.4f}")
                else:
                    print(f"  {feat}: Not enough data")
            else:
                print(f"  {feat}: Feature not found")
    else:
        print("Validation metadata not found. Skipping detailed failure analysis.")

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.7462618350982666

    print(f"\nChecking submission criteria: {final_metric} < {THRESHOLD}?")

    if final_metric < THRESHOLD:
        print("Criteria met. Generating submission...")
        trainer.predict()
    else:
        print("Criteria NOT met. Skipping submission generation.")


if __name__ == "__main__":
    main()
