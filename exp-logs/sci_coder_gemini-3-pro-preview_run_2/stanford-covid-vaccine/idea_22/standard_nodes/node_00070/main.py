import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import set_seed, MetricTracker
from library.loss import MaskedMCRMSELoss
from library.data import get_loader
from library.model import DecoupledDenseNet
from library.train import train_epoch, validate, generate_submission


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis by correlating model error with input features.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    # 1. Calculate per-sample error
    sample_errors = []
    sample_ids = []

    criterion = torch.nn.MSELoss(reduction="none")
    scored_indices = Config.SCORED_INDICES
    pred_len = Config.PRED_LEN

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            outputs = model(inputs, partner_indices)

            # Slice to scored length
            if outputs.shape[1] > pred_len:
                outputs = outputs[:, :pred_len, :]
            if targets.shape[1] > pred_len:
                targets = targets[:, :pred_len, :]

            # Select scored columns
            outputs_scored = outputs[:, :, scored_indices]
            targets_scored = targets[:, :, scored_indices]

            # Compute MSE per sample (average over seq_len and targets)
            # Shape: (Batch, Seq, Targets) -> (Batch,)
            mse = torch.mean((outputs_scored - targets_scored) ** 2, dim=(1, 2))
            rmse = torch.sqrt(mse)

            sample_errors.extend(rmse.cpu().numpy())
            sample_ids.extend(ids)

    # 2. Load Metadata
    val_df = pd.read_csv(Config.VAL_CSV)

    # Create a DataFrame for errors
    error_df = pd.DataFrame({"id": sample_ids, "rmse": sample_errors})

    # Merge with metadata
    merged_df = pd.merge(val_df, error_df, on="id")

    # 3. Feature Engineering for Correlation
    # Base counts
    merged_df["count_A"] = merged_df["sequence"].apply(lambda x: x.count("A"))
    merged_df["count_G"] = merged_df["sequence"].apply(lambda x: x.count("G"))
    merged_df["count_C"] = merged_df["sequence"].apply(lambda x: x.count("C"))
    merged_df["count_U"] = merged_df["sequence"].apply(lambda x: x.count("U"))

    # Features to check
    features = [
        "signal_to_noise",
        "mean_reactivity",
        "count_A",
        "count_G",
        "count_C",
        "count_U",
    ]

    print(f"Correlation with Validation RMSE (N={len(merged_df)}):")
    for feat in features:
        if feat in merged_df.columns:
            # Drop NaNs if any
            valid_data = merged_df[[feat, "rmse"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["rmse"])
                print(f"  {feat:<20}: {corr:.4f}")


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loaders
    # Using load_cached_data=True to speed up if cache exists
    train_loader = get_loader(split="train", shuffle=True, load_cached_data=True)
    val_loader = get_loader(split="val", shuffle=False, load_cached_data=True)

    # 3. Model Initialization
    model = DecoupledDenseNet().to(device)
    criterion = MaskedMCRMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 4. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_score = validate(model, val_loader, device)

        scheduler.step(val_score)

        # Save best model
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 5. Final Evaluation
    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Compute metric on full validation set
    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 7. Submission
    THRESHOLD = 0.5417620723771521

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission(model, device)

        # Move submission to requested directory
        target_dir = "./submission"
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, "submission.csv")

        # Config generates to Config.SUBMISSION_PATH
        source_path = Config.SUBMISSION_PATH

        if os.path.exists(source_path):
            shutil.move(source_path, target_path)
            print(f"Submission moved to {target_path}")
        else:
            print(f"Error: Submission file not found at {source_path}")
    else:
        print(
            f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
