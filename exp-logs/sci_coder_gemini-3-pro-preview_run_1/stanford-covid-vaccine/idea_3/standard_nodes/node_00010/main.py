import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, mcrmse_metric
from library.dataset import RNADataset
from library.engine import Engine
from library.model import masked_mse_loss


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Fast baseline settings
    FAST_EPOCHS = 10

    # 2. Training
    engine = Engine(device=device)
    engine.run_training(epochs=FAST_EPOCHS)

    # 3. Final Validation Assessment
    print("\nRunning Final Validation Assessment...")

    # Load best model
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError("Model checkpoint not found after training.")

    engine.model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    )
    engine.model.eval()

    val_dataset = RNADataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["sequence"].to(device)
            struct = batch["structure"].to(device)
            loop = batch["loop"].to(device)
            targets = batch["targets"].to(device)

            preds = engine.model(seq, struct, loop)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(batch["id"])

    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)

    # Calculate Final Metric
    final_metric = mcrmse_metric(all_targets_tensor, all_preds_tensor)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate per-sample error (MCRMSE for each sample)
    # We need to replicate the logic of mcrmse_metric but per sample
    seq_scored = Config.PRED_LEN
    scored_col_indices = [0, 1, 3]  # reactivity, deg_Mg_pH10, deg_Mg_50C

    y_pred_sliced = all_preds_tensor[:, :seq_scored, scored_col_indices]
    y_true_sliced = all_targets_tensor[:, :seq_scored, scored_col_indices]

    # MSE per sample: (N, 68, 3) -> (N,)
    # Mean over length and columns, then sqrt
    sample_mse = torch.mean((y_true_sliced - y_pred_sliced) ** 2, dim=(1, 2))
    sample_rmse = torch.sqrt(sample_mse).numpy()

    # Load metadata for features
    df_val = pd.read_parquet(Config.VAL_PATH)

    # Ensure alignment
    # The dataloader order is preserved because shuffle=False
    # But to be safe, we map errors to IDs
    error_df = pd.DataFrame({"id": all_ids, "error": sample_rmse})
    analysis_df = pd.merge(df_val, error_df, on="id")

    # Feature Engineering for Analysis
    analysis_df["len_A"] = analysis_df["sequence"].apply(lambda x: x.count("A"))
    analysis_df["len_G"] = analysis_df["sequence"].apply(lambda x: x.count("G"))
    analysis_df["len_C"] = analysis_df["sequence"].apply(lambda x: x.count("C"))
    analysis_df["len_U"] = analysis_df["sequence"].apply(lambda x: x.count("U"))

    features_to_check = ["signal_to_noise", "len_A", "len_G", "len_C", "len_U"]
    if "SN_filter" in analysis_df.columns:
        features_to_check.append("SN_filter")

    print("Correlation between Error Magnitude and Features:")
    for feat in features_to_check:
        if feat in analysis_df.columns:
            corr = analysis_df[feat].corr(analysis_df["error"])
            print(f"  {feat}: {corr:.4f}")

    # 5. Submission Logic
    THRESHOLD = 0.7462618350982666

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )
        engine.inference()
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
