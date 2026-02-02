import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from library.config import Config
from library.train import run_training
from library.model import RNAModel
from library.dataset import get_dataset, RNADataset
from library.utils import seed_everything, calculate_mcrmse


def analyze_failures(y_true, y_pred, df_val):
    """
    Performs failure analysis by correlating error magnitude with features.

    Args:
        y_true (np.ndarray): Ground truth targets (N, 68, 3).
        y_pred (np.ndarray): Predicted targets (N, 68, 3).
        df_val (pd.DataFrame): Validation metadata.
    """
    # Calculate MSE per sample (average over 68 positions and 3 channels)
    # Result shape: (N,)
    mse_per_sample = np.mean((y_true - y_pred) ** 2, axis=(1, 2))

    print("\nFailure Analysis (Correlation with MSE per sample):")

    # 1. Signal to Noise
    if "signal_to_noise" in df_val.columns:
        sn = df_val["signal_to_noise"].values
        # Ensure alignment
        if len(sn) == len(mse_per_sample):
            # Handle potential NaNs just in case
            valid_mask = ~np.isnan(sn)
            if valid_mask.sum() > 1:
                corr, _ = pearsonr(sn[valid_mask], mse_per_sample[valid_mask])
                print(f"  signal_to_noise: {corr:.10f}")

    # 2. Base Counts
    sequences = df_val["sequence"].values
    len_A = np.array([s.count("A") for s in sequences])
    len_G = np.array([s.count("G") for s in sequences])
    len_C = np.array([s.count("C") for s in sequences])
    len_U = np.array([s.count("U") for s in sequences])

    for name, data in [
        ("len_A", len_A),
        ("len_G", len_G),
        ("len_C", len_C),
        ("len_U", len_U),
    ]:
        if len(data) == len(mse_per_sample):
            corr, _ = pearsonr(data, mse_per_sample)
            print(f"  {name}: {corr:.10f}")

    # 3. Structure Counts
    structures = df_val["structure"].values
    paired = np.array([s.count("(") for s in structures])
    if len(paired) == len(mse_per_sample):
        corr, _ = pearsonr(paired, mse_per_sample)
        print(f"  paired_bases: {corr:.10f}")


def generate_submission(model, device):
    """
    Generates submission file for the test set.
    """
    print("Generating submission...")

    # Load Test Data
    test_ids, test_seq, test_loop, test_dist = get_dataset(
        Config.TEST_PATH, "test", load_cached_data=True
    )
    test_ds = RNADataset(test_seq, test_loop, test_dist)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    model.eval()
    all_preds = []

    with torch.no_grad():
        for x_seq, x_loop, x_dist in test_loader:
            x_seq = x_seq.to(device)
            x_loop = x_loop.to(device)
            x_dist = x_dist.to(device)

            # Predict full length (B, 107, 3)
            preds = model(x_seq, x_loop, x_dist)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 107, 3)

    # Prepare DataFrame
    submission_data = []

    # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Model outputs: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (2)
    # We must fill deg_pH10 and deg_50C with 0.0

    for i, sample_id in enumerate(test_ids):
        sample_preds = all_preds[i]  # (107, 3)

        for j in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{j}"
            reactivity = sample_preds[j, 0]
            deg_Mg_pH10 = sample_preds[j, 1]
            deg_pH10 = 0.0
            deg_Mg_50C = sample_preds[j, 2]
            deg_50C = 0.0

            submission_data.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    sub_df = pd.DataFrame(submission_data, columns=columns)

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Train Model
    # Using default Config.EPOCHS (20) which is fast enough for the dataset size
    best_model_path = run_training(load_cached_data=True)

    # 2. Load Best Model for Analysis
    device = Config.DEVICE
    model = RNAModel(Config).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # 3. Validation Inference
    # We re-run inference to ensure we have predictions corresponding exactly to the best model state
    val_ids, val_seq, val_loop, val_dist, val_y = get_dataset(
        Config.VAL_PATH, "val", load_cached_data=True
    )
    val_ds = RNADataset(val_seq, val_loop, val_dist, val_y)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_seq, x_loop, x_dist, y in val_loader:
            x_seq = x_seq.to(device)
            x_loop = x_loop.to(device)
            x_dist = x_dist.to(device)

            preds = model(x_seq, x_loop, x_dist)

            # Slice to scored length (68) for metric calculation
            preds_scored = preds[:, : Config.PRED_LEN, :]

            all_preds.append(preds_scored.cpu())
            all_targets.append(y)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 4. Calculate Metric
    metric = calculate_mcrmse(all_targets, all_preds)
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    df_val = pd.read_parquet(Config.VAL_PATH)
    analyze_failures(all_targets.numpy(), all_preds.numpy(), df_val)

    # 6. Submission Logic
    THRESHOLD = 0.6176461577
    if metric < THRESHOLD:
        generate_submission(model, device)
    else:
        print(f"Metric {metric} >= {THRESHOLD}. No submission generated.")


if __name__ == "__main__":
    main()
