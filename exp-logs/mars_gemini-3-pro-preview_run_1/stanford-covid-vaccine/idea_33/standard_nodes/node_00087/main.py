import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.engine import run_training, run_inference, evaluate
from library.dataset import get_dataset
from library.model import RNAModel
from library.utils import seed_everything


def main():
    # 1. Setup and Configuration
    config = Config()

    # Update submission path to match requirements
    config.SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs("./submission", exist_ok=True)

    # Ensure reproducibility
    seed_everything(config.SEED)

    print("==================================================")
    print("Step 1: Training Model")
    print("==================================================")

    # Run training
    # The engine handles the loop, saving the best model, and printing progress.
    # We use the full dataset as it is small (1728 samples) and fits within the time limit.
    best_model_path = run_training(config=config)

    print("\n==================================================")
    print("Step 2: Validation & Failure Analysis")
    print("==================================================")

    # Load the best model for analysis
    model = RNAModel(config).to(config.DEVICE)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    else:
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    model.eval()

    # Load validation dataset
    val_ds = get_dataset("val", config)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Compute Final Validation Metric
    final_metric = evaluate(model, val_loader, config.DEVICE, config)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")

    # 1. Collect predictions and targets for per-sample analysis
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(config.DEVICE)

            preds = model(batch)
            targets = batch["targets"]

            # Slice to scored region (first 68 bases)
            preds_scored = preds[:, : config.SEQ_SCORED, :]
            targets_scored = targets[:, : config.SEQ_SCORED, :]

            all_preds.append(preds_scored.cpu())
            all_targets.append(targets_scored.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 2. Calculate error per sample (RMSE averaged over the 3 targets)
    # Shape: (N_samples, Seq_Scored, 3)
    mse_per_sample = torch.mean((all_preds - all_targets) ** 2, dim=(1, 2))
    error_per_sample = torch.sqrt(mse_per_sample).numpy()

    # 3. Load metadata to correlate with error
    val_df = pd.read_parquet(config.VAL_PARQUET)

    # Ensure dataframe length matches predictions
    if len(val_df) != len(error_per_sample):
        print(
            f"Warning: Metadata length ({len(val_df)}) differs from prediction count ({len(error_per_sample)}). Skipping correlation."
        )
    else:
        val_df["model_error"] = error_per_sample

        # Extract features
        val_df["len_A"] = val_df["sequence"].apply(lambda x: x.count("A"))
        val_df["len_G"] = val_df["sequence"].apply(lambda x: x.count("G"))
        val_df["len_C"] = val_df["sequence"].apply(lambda x: x.count("C"))
        val_df["len_U"] = val_df["sequence"].apply(lambda x: x.count("U"))
        val_df["paired_bases"] = val_df["structure"].apply(lambda x: x.count("("))

        analysis_features = [
            "signal_to_noise",
            "SN_filter",
            "len_A",
            "len_G",
            "len_C",
            "len_U",
            "paired_bases",
        ]

        print("Correlation between Model Error and Input Features:")
        for feat in analysis_features:
            if feat in val_df.columns:
                if pd.api.types.is_numeric_dtype(val_df[feat]):
                    corr = val_df[feat].corr(val_df["model_error"])
                    print(f"  {feat}: {corr:.10f}")

    print("\n==================================================")
    print("Step 3: Submission Generation")
    print("==================================================")

    threshold = 0.6199890971183777

    if final_metric < threshold:
        print(f"Metric {final_metric} is better than threshold {threshold}.")
        print("Generating submission file...")
        run_inference(config)
    else:
        print(f"Metric {final_metric} is NOT better than threshold {threshold}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
