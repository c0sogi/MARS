import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_all, get_device, mcrmse
from library.dataset import get_dataloader
from library.model import RNAModel, generate_submission
from library.engine import train_fn, eval_fn


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis by correlating sample-wise error with input features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    # 1. Collect Predictions and Targets per sample
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop_type"].to(device)
            dist = batch["pairing_distance"].to(device)
            target = batch["target"].to(device)
            batch_ids = batch["id"]

            pred = model(seq, loop, dist)

            # Slice to scored region
            seq_scored = target.shape[1]
            pred_scored = pred[:, :seq_scored, :]

            all_preds.append(pred_scored.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_ids.extend(batch_ids)

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 68, 3)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 68, 3)

    # 2. Calculate RMSE per sample
    # Error = sqrt(mean((y - y_hat)^2)) over the 68 positions and 3 channels
    squared_diff = (all_targets - all_preds) ** 2
    # Mean over seq_scored (axis 1) and channels (axis 2)
    mse_per_sample = np.mean(squared_diff, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Create a DataFrame for analysis
    df_error = pd.DataFrame({"id": all_ids, "rmse": rmse_per_sample})

    # 3. Load Metadata for features
    # We read the validation parquet file to get features like signal_to_noise, sequence, etc.
    df_val_meta = pd.read_parquet(Config.VAL_FILE)

    # Merge error data with metadata
    df_analysis = pd.merge(df_error, df_val_meta, on="id", how="inner")

    # 4. Feature Engineering for Correlation
    # GC Content
    df_analysis["gc_content"] = df_analysis["sequence"].apply(
        lambda x: (x.count("G") + x.count("C")) / len(x)
    )
    # Structure Density (Paired bases count)
    df_analysis["structure_density"] = df_analysis["structure"].apply(
        lambda x: (x.count("(") + x.count(")")) / len(x)
    )
    # Ensure signal_to_noise is numeric
    if "signal_to_noise" in df_analysis.columns:
        df_analysis["signal_to_noise"] = pd.to_numeric(
            df_analysis["signal_to_noise"], errors="coerce"
        ).fillna(0)

    # 5. Compute Correlations
    features = ["signal_to_noise", "gc_content", "structure_density", "seq_length"]
    print(f"{'Feature':<20} | {'Correlation with Error (RMSE)':<30}")
    print("-" * 55)

    for feat in features:
        if feat in df_analysis.columns:
            # Check if feature has variance
            if df_analysis[feat].nunique() > 1:
                corr, _ = pearsonr(df_analysis[feat], df_analysis["rmse"])
                print(f"{feat:<20} | {corr:.6f}")
            else:
                print(f"{feat:<20} | N/A (Constant value)")
        else:
            print(f"{feat:<20} | Not found in metadata")
    print("-" * 55)


def main():
    # 1. Setup
    config = Config()
    # Restore full epoch budget for convergence (Cite solution_lesson_node_00131)
    config.EPOCHS = 20

    seed_all(config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading Data...")
    train_loader = get_dataloader(
        "train", batch_size=config.BATCH_SIZE, shuffle=True, load_cached_data=True
    )
    val_loader = get_dataloader(
        "val", batch_size=config.BATCH_SIZE, shuffle=False, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = RNAModel(config).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # 4. Training Loop
    best_score = float("inf")
    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        train_loss = train_fn(
            model, train_loader, optimizer, device, clip_grad=config.CLIP_GRAD
        )
        val_score = eval_fn(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), config.MODEL_PATH)

    print("Training complete.")

    # 5. Final Evaluation
    # Load best model
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))

    # Compute final metric on full validation set
    final_val_metric = eval_fn(model, val_loader, device)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_val_metric}")

    # 6. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 7. Conditional Submission
    THRESHOLD = 0.6176461577
    if final_val_metric < THRESHOLD:
        print(
            f"Validation metric ({final_val_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, config, device)
    else:
        print(
            f"Validation metric ({final_val_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
