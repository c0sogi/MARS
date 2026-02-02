import os
import sys
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.trainer import Trainer
from library.inference import InferencePipeline
from library.model import ResUNet1D
from library.dataset import gnss_collate_fn, GnssSequenceDataset
from library.data_processing import GNSSPreprocessor
from torch.utils.data import DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_competition_metric(errors_df):
    """
    Calculates the competition metric:
    Mean of the (50th percentile + 95th percentile) / 2 for each phone.
    """
    phone_scores = []
    for phone, group in errors_df.groupby("phone_name"):
        errors = group["error"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        score = (p50 + p95) / 2
        phone_scores.append(score)

    if not phone_scores:
        return 0.0
    return np.mean(phone_scores)


def run_validation_and_analysis(model, val_loader, device):
    print("\n--- Running Validation and Failure Analysis ---")
    model.eval()

    results = []

    # Feature indices for analysis (based on Config.IN_CHANNELS = 26)
    # 0-3: S1_Cn0 (mean, std, min, max)
    # 4-7: S1_Elev
    # ...
    # We will track mean Cn0 (idx 0) and mean Elevation (idx 4) of Stratum 1 (Global)
    # Note: Features are normalized in dataset, but correlations work fine.

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # Prepare inputs
            # features: (B, T, C) -> Model expects (B, C, T)
            features = batch["features"].to(device).transpose(1, 2)
            targets = batch["targets"].to(device)  # (B, T, 2)
            mask = batch["mask"].to(device)  # (B, T)
            phone_names = batch["phone_name"]

            # Forward pass
            final_out, _ = model(features)
            final_out = final_out.transpose(1, 2)  # (B, T, 2)

            # Calculate Errors
            # Truncate if necessary (though usually aligned)
            seq_len = final_out.size(1)
            targets = targets[:, :seq_len, :]
            mask = mask[:, :seq_len]

            # Euclidean distance in meters
            diff = final_out - targets
            dist = torch.sqrt(torch.sum(diff**2, dim=2))  # (B, T)

            # Extract data for analysis
            # features is (B, C, T). Transpose back to (B, T, C) for indexing
            feat_cpu = features.transpose(1, 2).cpu().numpy()
            dist_cpu = dist.cpu().numpy()
            mask_cpu = mask.cpu().numpy()

            batch_size = dist.shape[0]

            for i in range(batch_size):
                # Get valid indices
                valid_indices = mask_cpu[i]

                if not np.any(valid_indices):
                    continue

                valid_errors = dist_cpu[i][valid_indices]
                valid_feats = feat_cpu[i][valid_indices]

                # S1_Cn0_mean is index 0, S1_Elev_mean is index 4
                s1_cn0 = valid_feats[:, 0]
                s1_elev = valid_feats[:, 4]

                phone = phone_names[i]

                for err, cn0, elev in zip(valid_errors, s1_cn0, s1_elev):
                    results.append(
                        {
                            "phone_name": phone,
                            "error": err,
                            "s1_cn0_mean": cn0,
                            "s1_elev_mean": elev,
                        }
                    )

    df_results = pd.DataFrame(results)

    # 1. Calculate Metric
    final_metric = calculate_competition_metric(df_results)
    print(f"Final Validation Metric: {final_metric}")

    # 2. Failure Analysis
    print("\n--- Failure Analysis (Correlation with Error) ---")
    if not df_results.empty:
        corr_cn0, _ = pearsonr(df_results["error"], df_results["s1_cn0_mean"])
        corr_elev, _ = pearsonr(df_results["error"], df_results["s1_elev_mean"])

        print(f"Correlation (Error vs S1_Cn0_Mean): {corr_cn0:.4f}")
        print(f"Correlation (Error vs S1_Elev_Mean): {corr_elev:.4f}")

        if corr_cn0 < -0.1:
            print("Observation: Lower signal strength correlates with higher error.")
        if corr_elev < -0.1:
            print("Observation: Lower elevation correlates with higher error.")
    else:
        print("No validation results to analyze.")

    return final_metric


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # Modify Config for fast baseline execution within time limit
    Config.EPOCHS = 5  # Reduced epochs for speed
    Config.BATCH_SIZE = 16  # Increase batch size slightly for speed if memory allows

    print(f"Running with EPOCHS={Config.EPOCHS}, BATCH_SIZE={Config.BATCH_SIZE}")

    # 2. Training
    trainer = Trainer()
    # Assuming preprocessed data exists in ./working/idea_27/ or similar from previous steps
    # If not, it will generate it. We set load_cached_data=True to use existing if available.
    # Note: The provided file structure shows parquet files in ./working, so we should try to use them.
    # We need to ensure Config points to the right place. The provided Config points to ./working/idea_27.
    # We will rely on Trainer to handle data loading.

    print("Starting Training...")
    trainer.fit(load_cached_data=True)

    # 3. Validation & Analysis
    # Load the best model
    model = ResUNet1D().to(Config.DEVICE)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )
        print(f"Loaded best model from {Config.MODEL_SAVE_PATH}")
    else:
        print("Warning: Model save path not found, using current model state.")

    # Get Validation Loader
    # We reuse the preprocessor logic but need to instantiate dataset manually or via helper
    preprocessor = GNSSPreprocessor()
    val_df = preprocessor.process_val_data(load_cached_data=True)
    val_dataset = GnssSequenceDataset(val_df, is_test=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=gnss_collate_fn,
        pin_memory=True,
    )

    metric = run_validation_and_analysis(model, val_loader, Config.DEVICE)

    # 4. Submission
    THRESHOLD = 3.7864967500302016
    if metric < THRESHOLD:
        print(f"\nMetric {metric} < {THRESHOLD}. Generating submission...")
        pipeline = InferencePipeline()
        pipeline.run(load_cached_data=True)
    else:
        print(f"\nMetric {metric} >= {THRESHOLD}. Skipping submission generation.")


if __name__ == "__main__":
    main()
