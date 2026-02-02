import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.stats

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, haversine_distance, enu_to_latlon
from library.data_processing import get_data
from library.dataset import get_datasets, FEATURE_COLUMNS
from library.model import AttentionGatedResUNet1D
from library.loss import MultiScaleMAELoss
from library.trainer import Trainer, generate_submission

# Initialize Logger
logger = get_logger()


def calculate_competition_metric(model, val_loader, device):
    """
    Calculates the competition metric: Mean of (50th + 95th percentile errors) across phones.
    Also returns a DataFrame with point-wise errors and features for failure analysis.
    """
    model.eval()
    all_errors = []
    all_phones = []
    all_features = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device, dtype=torch.float32)
            mask = batch["mask"].to(device, dtype=torch.float32)
            # We only need the high-res target for evaluation
            targets = batch["targets"][0].to(device, dtype=torch.float32)

            # Forward pass (get highest resolution output)
            outputs_list = model(features)
            predictions = outputs_list[0]  # (B, 2, L)

            # Dimensions
            B, C, L = predictions.shape

            # Convert to numpy for processing
            pred_np = predictions.cpu().numpy()
            targ_np = targets.cpu().numpy()
            mask_np = mask.cpu().numpy()
            feat_np = features.cpu().numpy()

            # Iterate through batch
            for i in range(B):
                # Extract valid length using mask
                valid_len = int(mask_np[i].sum())
                if valid_len == 0:
                    continue

                # Slicing valid data
                # Shape: (2, valid_len) -> (valid_len, 2)
                p_seq = pred_np[i, :, :valid_len].T
                t_seq = targ_np[i, :, :valid_len].T
                f_seq = feat_np[i, :, :valid_len].T

                # Calculate Euclidean distance in meters (ENU space)
                # Error vector = Predicted - Target
                # Distance = sqrt(dE^2 + dN^2)
                diff = p_seq - t_seq
                dists = np.sqrt(np.sum(diff**2, axis=1))

                phone_name = batch["meta"]["phone_name"][i]

                all_errors.extend(dists)
                all_phones.extend([phone_name] * valid_len)
                all_features.append(f_seq)

    # Concatenate all data
    if not all_errors:
        return float("inf"), pd.DataFrame()

    all_errors = np.array(all_errors)
    all_phones = np.array(all_phones)
    all_features = np.concatenate(all_features, axis=0)

    # Create DataFrame for analysis
    eval_df = pd.DataFrame({"phone_name": all_phones, "error": all_errors})

    # Add features to DataFrame for failure analysis
    for idx, col_name in enumerate(FEATURE_COLUMNS):
        eval_df[col_name] = all_features[:, idx]

    # Calculate Metric
    # Group by phone
    phone_metrics = []
    unique_phones = np.unique(all_phones)

    for phone in unique_phones:
        phone_errors = eval_df[eval_df["phone_name"] == phone]["error"].values
        p50 = np.percentile(phone_errors, 50)
        p95 = np.percentile(phone_errors, 95)
        avg_metric = (p50 + p95) / 2.0
        phone_metrics.append(avg_metric)

    final_metric = np.mean(phone_metrics)

    return final_metric, eval_df


def perform_failure_analysis(eval_df):
    """
    Correlates prediction error with input features.
    """
    logger.info("Performing Failure Analysis...")

    # Calculate correlation between 'error' and features
    # We use Spearman correlation to capture monotonic relationships (robust to outliers)
    correlations = {}
    for col in FEATURE_COLUMNS:
        if col in eval_df.columns:
            corr, _ = scipy.stats.spearmanr(eval_df["error"], eval_df[col])
            correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print(
        "\n[Failure Analysis] Correlation between Error Magnitude and Input Features:"
    )
    print(f"{'Feature':<40} | {'Spearman Corr':<10}")
    print("-" * 55)
    for feat, corr in sorted_corr:
        print(f"{feat:<40} | {corr:.4f}")
    print("-" * 55)


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Load Data
    # load_cached_data=True will look for parquet files in ./working/idea_15/cache
    train_df, val_df, test_df = get_data(load_cached_data=True)

    if train_df.empty:
        logger.error("Training data not found or empty.")
        return

    # 3. Subsample Training Data for Fast Baseline
    # We select 25% of the drives to speed up the epoch time
    unique_drives = train_df["drive_id"].unique()
    # Ensure we select at least one drive if few exist
    n_sample = max(1, int(len(unique_drives) * 0.25))
    sampled_drives = np.random.choice(unique_drives, n_sample, replace=False)

    logger.info(
        f"Subsampling training data: Using {len(sampled_drives)}/{len(unique_drives)} drives."
    )
    train_df_sampled = train_df[train_df["drive_id"].isin(sampled_drives)].copy()

    # 4. Create Datasets & Loaders
    # get_datasets calculates normalization stats from train_df_sampled and applies to others
    train_dataset, val_dataset, test_dataset = get_datasets(
        train_df_sampled, val_df, test_df
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Initialize Model & Training Components
    model = AttentionGatedResUNet1D().to(device)
    criterion = MultiScaleMAELoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # 6. Train
    # Limiting epochs to 5 for fast baseline execution
    EPOCHS = 5
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=3,  # Strict early stopping
    )

    trainer.fit(train_loader, val_loader, epochs=EPOCHS)

    # 7. Load Best Model for Evaluation
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        logger.warning("No model checkpoint found. Using current model state.")

    # 8. Calculate Validation Metric
    logger.info("Calculating final validation metric...")
    val_metric, eval_df = calculate_competition_metric(model, val_loader, device)

    print(f"Final Validation Metric: {val_metric}")

    # 9. Failure Analysis
    if not eval_df.empty:
        perform_failure_analysis(eval_df)

    # 10. Generate Submission
    THRESHOLD = 3.802240262877392
    if val_metric < THRESHOLD:
        logger.info(
            f"Validation metric ({val_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        logger.info(
            f"Validation metric ({val_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
