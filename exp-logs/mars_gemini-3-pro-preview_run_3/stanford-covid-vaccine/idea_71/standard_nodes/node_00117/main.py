import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

# Ensure library can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, MetricCalculator
from library.data import get_dataloaders
from library.model import HighCapacityAugmentedBiGRU
from library.train import train_one_epoch, validate, inference


def main():
    # 1. Configuration
    # Using 15 epochs for a fast baseline execution as requested
    config = Config(epochs=15, batch_size=32)
    set_seed(config.SEED)

    # 2. Data Loading
    # load_cached_data=True ensures we use pre-processed .npz files from working directory
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # 3. Model Initialization
    model = HighCapacityAugmentedBiGRU(config)
    model.to(config.DEVICE)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    metric_calc = MetricCalculator(config)

    # 5. Training Loop
    best_score = float("inf")
    best_model_state = None

    for epoch in range(config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, metric_calc, config.DEVICE, config
        )

        # Validate
        val_score = validate(model, val_loader, metric_calc, config.DEVICE)

        # Update Scheduler
        scheduler.step()

        # Checkpoint
        if val_score < best_score:
            best_score = val_score
            best_model_state = model.state_dict()
            torch.save(best_model_state, config.MODEL_PATH)

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 6. Final Validation Metric
    final_val_score = validate(model, val_loader, metric_calc, config.DEVICE)
    print(f"Final Validation Metric: {final_val_score}")

    # 7. Failure Analysis
    print("Performing Failure Analysis...")

    # Load validation metadata
    val_df = pd.read_parquet(config.VAL_PATH)

    model.eval()
    sample_errors = []
    sample_ids = []

    # Indices for scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_indices = [config.TARGET_COLS.index(col) for col in config.SCORED_COLS]

    with torch.no_grad():
        for batch in val_loader:
            sequence = batch["sequence"].to(config.DEVICE)
            bpp_indices = batch["bpp_indices"].to(config.DEVICE)
            pair_mask = batch["pair_mask"].to(config.DEVICE)
            targets = batch["targets"].to(config.DEVICE)
            ids = batch["id"]

            # Forward pass
            preds = model(sequence, bpp_indices, pair_mask)

            # Slice predictions to scored length (68)
            preds_sliced = preds[:, : config.SEQ_SCORED, :]

            # Compute Squared Error
            diff = preds_sliced - targets
            squared_error = diff**2

            # Select only scored columns for the metric
            squared_error = squared_error[:, :, scored_indices]

            # Compute MCRMSE per sample
            # Mean over sequence (dim 1) -> MSE per column
            mse_per_col = torch.mean(squared_error, dim=1)
            # Sqrt -> RMSE per column
            rmse_per_col = torch.sqrt(mse_per_col)
            # Mean over columns -> MCRMSE per sample
            mcrmse_per_sample = torch.mean(rmse_per_col, dim=1)

            sample_errors.extend(mcrmse_per_sample.cpu().tolist())
            sample_ids.extend(ids)

    # Create analysis DataFrame
    error_df = pd.DataFrame({"id": sample_ids, "error": sample_errors})
    analysis_df = pd.merge(val_df, error_df, on="id")

    # Feature Engineering for Analysis
    analysis_df["gc_content"] = analysis_df["sequence"].apply(
        lambda x: (x.count("G") + x.count("C")) / len(x)
    )

    # Calculate Correlations
    features = ["signal_to_noise", "SN_filter", "gc_content"]
    print("Correlation between Error and Features:")
    for feat in features:
        if feat in analysis_df.columns:
            corr = analysis_df["error"].corr(analysis_df[feat])
            print(f"{feat}: {corr}")

    # 8. Conditional Submission
    THRESHOLD = 0.5884495377540588

    if final_val_score < THRESHOLD:
        preds, test_ids = inference(model, test_loader, config.DEVICE)

        submission_rows = []
        for i, sample_id in enumerate(test_ids):
            sample_preds = preds[i]  # Shape (107, 5)
            for seqpos in range(config.SEQ_LENGTH):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_preds[seqpos].tolist()
                submission_rows.append([row_id] + row_values)

        submission_df = pd.DataFrame(
            submission_rows, columns=["id_seqpos"] + config.TARGET_COLS
        )

        # Save submission
        os.makedirs("./submission", exist_ok=True)
        submission_path = "./submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
