import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.train import train_model, predict_and_submit
from library.dataset import RNADataset
from library.model import RNAMultiTaskBiGRU
from library.utils import set_seed

# ==================================================================================
# CONFIGURATION
# ==================================================================================


class RunConfig(Config):
    """
    Configuration for the execution run.
    Adjusts epochs for a fast baseline while maintaining model capacity.
    """

    def __init__(self):
        super().__init__()
        self.EPOCHS = 15  # Reduced from 25 for speed constraint
        self.WORKING_DIR = "./working/execution"
        self.SUBMISSION_PATH = "./submission/submission.csv"
        self.BATCH_SIZE = 32
        self.NUM_LAYERS = 5  # Keep deep architecture
        self.HIDDEN_DIM = 256
        self.DROPOUT = 0.1
        self.LR = 1e-3
        self.SEED = 42


# ==================================================================================
# FAILURE ANALYSIS
# ==================================================================================


def perform_failure_analysis(config):
    """
    Analyzes the correlation between model error and input features on the validation set.
    """
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Validation Data
    val_dataset = RNADataset("val", config=config)
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 2. Load Best Model
    model = RNAMultiTaskBiGRU(config).to(device)
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print("Error: Best model not found for failure analysis.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Run Inference
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            pair_idx = batch["pair_idx"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            reg_out, _ = model(seq, loop, pair_idx, pair_dist)

            # Select scored columns and length
            scored_preds = reg_out[:, : config.SCORED_LEN, :]
            scored_targets = targets[:, : config.SCORED_LEN, config.SCORED_INDICES]

            all_preds.append(scored_preds.cpu().numpy())
            all_targets.append(scored_targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 68, 3)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 68, 3)

    # 4. Calculate Sample-wise RMSE
    # Mean over columns (axis 2) and sequence positions (axis 1)
    # Error definition: RMSE per sample
    mse_per_sample = np.mean((all_preds - all_targets) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 5. Load Metadata for Correlation
    meta_df = pd.read_parquet(os.path.join(config.METADATA_DIR, "val.parquet"))

    # Ensure alignment (dataset loads in order of dataframe)
    if len(meta_df) != len(rmse_per_sample):
        print("Warning: Metadata length mismatch. Skipping analysis.")
        return

    # 6. Construct Analysis DataFrame
    analysis_df = pd.DataFrame(
        {
            "error": rmse_per_sample,
            "signal_to_noise": (
                meta_df["signal_to_noise"]
                if "signal_to_noise" in meta_df.columns
                else 0
            ),
            "SN_filter": meta_df["SN_filter"] if "SN_filter" in meta_df.columns else 0,
            "len_A": meta_df["sequence"].apply(lambda x: x.count("A")),
            "len_G": meta_df["sequence"].apply(lambda x: x.count("G")),
            "len_C": meta_df["sequence"].apply(lambda x: x.count("C")),
            "len_U": meta_df["sequence"].apply(lambda x: x.count("U")),
        }
    )

    # 7. Compute Correlations
    print(f"Analyzing {len(analysis_df)} validation samples...")
    correlations = analysis_df.corr(method="pearson")["error"].drop("error")

    print("\nCorrelation with Model Error (RMSE):")
    for feat, corr in correlations.sort_values(ascending=False).items():
        print(f"  {feat:15s}: {corr:.4f}")


# ==================================================================================
# MAIN EXECUTION
# ==================================================================================


def main():
    # 1. Setup
    config = RunConfig()
    set_seed(config.SEED)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"Running with configuration:")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # 2. Training
    # train_model returns the best validation MCRMSE
    best_mcrmse = train_model(config)

    # 3. Report Metric
    # Required format: Final Validation Metric: <value>
    print(f"Final Validation Metric: {best_mcrmse}")

    # 4. Failure Analysis
    try:
        perform_failure_analysis(config)
    except Exception as e:
        print(f"Failure analysis failed: {e}")

    # 5. Submission
    THRESHOLD = 0.6226052641868591
    if best_mcrmse < THRESHOLD:
        print(f"\nMetric {best_mcrmse:.6f} < {THRESHOLD:.6f}. Generating submission...")
        predict_and_submit(config)
    else:
        print(f"\nMetric {best_mcrmse:.6f} >= {THRESHOLD:.6f}. Skipping submission.")


if __name__ == "__main__":
    main()
