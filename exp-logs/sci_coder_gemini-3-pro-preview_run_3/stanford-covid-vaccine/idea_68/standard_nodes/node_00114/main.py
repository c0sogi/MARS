import os
import warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.utils import seed_everything, metric_mcrmse_scored
from library.data import process_dataframe, RNADataset
from library.train import train_model
from library.model import RNAModel

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Fast Baseline: Reduce epochs to ensure quick execution while maintaining full data coverage
    # The high-capacity model needs data, so we don't subsample the dataset, just training time.
    Config.EPOCHS = 15

    print("Starting execution...")

    # 2. Training
    # train_model handles data loading, model init, training loop, and saving best model
    model = train_model(config=Config)

    # 3. Validation & Metric Calculation
    print("Running validation inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load validation data manually to have access to the dataframe for failure analysis
    val_df = pd.read_parquet(Config.VAL_META)
    v_feat, v_pidx, v_pmask, v_targ, v_ids = process_dataframe(
        val_df, "val_data", load_cached_data=True
    )
    val_ds = RNADataset(v_feat, v_pidx, v_pmask, v_targ, v_ids)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            p_idx = batch["pair_indices"].to(device)
            p_mask = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)

            preds = model(features, p_idx, p_mask)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute Metric
    val_metric = metric_mcrmse_scored(all_preds, all_targets, Config.SEQ_SCORED)
    print(f"Final Validation Metric: {val_metric}")

    # 4. Failure Analysis
    print("Performing failure analysis...")

    # Calculate per-sample error
    # Slice to scored length
    preds_sliced = all_preds[:, : Config.SEQ_SCORED, :]
    targets_sliced = all_targets[:, : Config.SEQ_SCORED, :]

    # Select scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    sel_indices = [0, 1, 3]
    p_sel = preds_sliced[:, :, sel_indices]
    t_sel = targets_sliced[:, :, sel_indices]

    # MSE per column per sample: (N, 3)
    # Mean over sequence length (dim 1)
    mse_per_col_sample = torch.mean((p_sel - t_sel) ** 2, dim=1)

    # RMSE per column per sample: (N, 3)
    rmse_per_col_sample = torch.sqrt(mse_per_col_sample)

    # Mean over columns: (N,) -> This is the MCRMSE for each sample
    error_per_sample = torch.mean(rmse_per_col_sample, dim=1).numpy()

    # Add to dataframe
    val_df["error"] = error_per_sample

    # Feature Engineering for Correlation
    # Sequence content
    val_df["pct_A"] = val_df["sequence"].apply(lambda s: s.count("A") / len(s))
    val_df["pct_G"] = val_df["sequence"].apply(lambda s: s.count("G") / len(s))
    val_df["pct_U"] = val_df["sequence"].apply(lambda s: s.count("U") / len(s))
    val_df["pct_C"] = val_df["sequence"].apply(lambda s: s.count("C") / len(s))

    # Correlations
    features_to_check = [
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_U",
        "pct_C",
    ]
    for feat in features_to_check:
        if feat in val_df.columns:
            corr = val_df["error"].corr(val_df[feat])
            print(f"Correlation between Error and {feat}: {corr:.4f}")

    # 5. Submission
    THRESHOLD = 0.5884495377540588
    if val_metric < THRESHOLD:
        print("Validation metric meets threshold. Generating submission...")

        # Load Test Data
        test_df = pd.read_parquet(Config.TEST_META)
        te_feat, te_pidx, te_pmask, te_targ, te_ids = process_dataframe(
            test_df, "test_data", load_cached_data=True
        )
        test_ds = RNADataset(te_feat, te_pidx, te_pmask, te_targ, te_ids)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_preds = []
        test_ids_list = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(device)
                p_idx = batch["pair_indices"].to(device)
                p_mask = batch["pair_masks"].to(device)
                ids = batch["id"]

                preds = model(features, p_idx, p_mask)  # (B, 107, 5)
                test_preds.append(preds.cpu().numpy())
                test_ids_list.extend(ids)

        test_preds = np.concatenate(test_preds, axis=0)

        # Format Submission
        submission_rows = []
        for i, sample_id in enumerate(test_ids_list):
            preds_i = test_preds[i]  # (107, 5)
            for pos in range(Config.SEQ_LEN):
                id_seqpos = f"{sample_id}_{pos}"
                # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
                # Note: Config.TARGET_COLS matches the required output order
                row = [id_seqpos] + preds_i[pos].tolist()
                submission_rows.append(row)

        cols = ["id_seqpos"] + Config.TARGET_COLS
        sub_df = pd.DataFrame(submission_rows, columns=cols)

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric {val_metric} is not lower than {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
