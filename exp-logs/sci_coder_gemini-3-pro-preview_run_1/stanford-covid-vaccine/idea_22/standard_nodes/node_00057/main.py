import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from library.config import Config, set_seed
from library.train import train
from library.model import RNAModel
from library.dataset import get_dataloaders
from library.utils import mcrmse_metric
from library.predict import generate_submission


def main():
    # 1. Setup and Configuration
    # Adjust epochs for a fast baseline run as requested
    Config.EPOCHS = 10
    set_seed(Config.SEED)

    print(f"Initializing run with Experiment: {Config.EXPERIMENT_NAME}")
    print(f"Device: {Config.DEVICE}")

    # 2. Training Phase
    print("\n=== Starting Training ===")
    # Train the model. This handles data loading, training loop, and saving the best model.
    # We explicitly pass epochs to override the default if necessary, though Config.EPOCHS was set above.
    train(epochs=Config.EPOCHS, load_cached_data=True)

    # 3. Validation Phase
    print("\n=== Starting Validation Evaluation ===")
    device = torch.device(Config.DEVICE)

    # Load the best model saved during training
    model = RNAModel(config=Config)
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Get Validation DataLoader
    # We use the same batch size as training or default
    _, val_loader = get_dataloaders(load_cached_data=True, batch_size=Config.BATCH_SIZE)

    all_preds = []
    all_targets = []
    all_ids = []

    # Inference loop on Validation set
    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            target = batch["target"].to(device)
            ids = batch["id"]

            outputs = model(seq, loop, dist)

            # Slice outputs and targets to the scored length (68)
            preds_slice = outputs[:, : Config.PRED_LEN, :].cpu().numpy()
            targets_slice = target[:, : Config.PRED_LEN, :].cpu().numpy()

            all_preds.append(preds_slice)
            all_targets.append(targets_slice)
            all_ids.extend(ids)

    # Concatenate results
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE
    final_metric = mcrmse_metric(y_true, y_pred)

    # Print metric in the required format
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate MSE per sample (average over sequence length and targets)
    # Shape: (N_samples, 68, 3)
    mse_per_sample = np.mean((y_true - y_pred) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load validation metadata to get features
    if os.path.exists(Config.VAL_DATA_PATH):
        val_df = pd.read_parquet(Config.VAL_DATA_PATH)

        # Create a dataframe for analysis
        analysis_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

        # Merge with metadata
        analysis_df = pd.merge(analysis_df, val_df, on="id", how="inner")

        # Generate derived features for correlation
        analysis_df["len_A"] = analysis_df["sequence"].apply(lambda x: x.count("A"))
        analysis_df["len_G"] = analysis_df["sequence"].apply(lambda x: x.count("G"))
        analysis_df["len_C"] = analysis_df["sequence"].apply(lambda x: x.count("C"))
        analysis_df["len_U"] = analysis_df["sequence"].apply(lambda x: x.count("U"))

        # List of features to check correlation with
        features_to_check = ["signal_to_noise", "len_A", "len_G", "len_C", "len_U"]
        if "SN_filter" in analysis_df.columns:
            features_to_check.append("SN_filter")

        print("Correlation between Error (RMSE) and Features:")
        for feat in features_to_check:
            if feat in analysis_df.columns:
                # Drop NaNs to ensure pearsonr works
                valid_data = analysis_df[[feat, "error"]].dropna()
                if len(valid_data) > 1:
                    corr, _ = pearsonr(valid_data[feat], valid_data["error"])
                    print(f"  {feat}: {corr:.4f}")
    else:
        print("Validation metadata not found. Skipping detailed failure analysis.")

    # 5. Submission Generation
    print("\n=== Submission Check ===")
    THRESHOLD = 0.6209375959946717

    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} is lower than threshold {THRESHOLD}.")
        print("Generating submission file...")
        generate_submission(load_cached_data=True)
    else:
        print(f"Metric {final_metric} is NOT lower than threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
