import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.data import get_dataloaders
from library.model import DeepResBiGRU
from library.train import train_one_epoch, validate
from library.inference import generate_submission


def main():
    # ------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------
    # Adjust Config for fast baseline execution
    Config.EPOCHS = 50
    Config.BATCH_SIZE = 64

    # Ensure output directories exist
    os.makedirs("./submission", exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)
    device = Config.DEVICE

    # ------------------------------------------------------------------
    # 2. Data Loading
    # ------------------------------------------------------------------
    # Load cached data if available, otherwise preprocess
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # ------------------------------------------------------------------
    # 3. Model Initialization
    # ------------------------------------------------------------------
    model = DeepResBiGRU().to(device)

    # ------------------------------------------------------------------
    # 4. Optimization Setup
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Loss for training (all columns) and validation (scored columns only)
    criterion_train = MCRMSELoss(select_columns=None)
    criterion_val = MCRMSELoss(select_columns=Config.SCORED_TARGET_INDICES)

    # ------------------------------------------------------------------
    # 5. Training Loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_train, device
        )
        val_loss = validate(model, val_loader, criterion_val, device)
        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # ------------------------------------------------------------------
    # 6. Final Evaluation
    # ------------------------------------------------------------------
    # Load best model weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

    # Compute final validation metric
    final_val_loss = validate(model, val_loader, criterion_val, device)
    print(f"Final Validation Metric: {final_val_loss}")

    # ------------------------------------------------------------------
    # 7. Failure Analysis
    # ------------------------------------------------------------------
    print("\nPerforming failure analysis...")

    # Load validation metadata for feature correlation
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)

    # Generate predictions on validation set
    model.eval()
    all_preds = []
    all_targets = []

    # We need to ensure we iterate in the same order as the DataFrame
    # get_dataloaders returns val_loader with shuffle=False, so order is preserved.
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Slice to prediction length (68) and scored columns
            # Outputs: (B, 107, 5) -> (B, 68, 3)
            outputs_sliced = outputs[:, : Config.PRED_LEN, Config.SCORED_TARGET_INDICES]
            targets_sliced = targets[:, : Config.PRED_LEN, Config.SCORED_TARGET_INDICES]

            all_preds.append(outputs_sliced.cpu().numpy())
            all_targets.append(targets_sliced.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate RMSE per sample
    # Shape: (N_samples, 68, 3) -> Mean over (68, 3) -> Sqrt
    mse_per_sample = np.mean((all_preds - all_targets) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Add error to dataframe
    # Ensure lengths match
    if len(rmse_per_sample) == len(val_df):
        val_df["error"] = rmse_per_sample

        # Feature Engineering for correlation
        val_df["pct_A"] = val_df["sequence"].apply(lambda x: x.count("A") / len(x))
        val_df["pct_G"] = val_df["sequence"].apply(lambda x: x.count("G") / len(x))
        val_df["pct_U"] = val_df["sequence"].apply(lambda x: x.count("U") / len(x))
        val_df["pct_C"] = val_df["sequence"].apply(lambda x: x.count("C") / len(x))

        # Calculate correlations
        corr_cols = ["signal_to_noise", "SN_filter", "pct_A", "pct_G", "pct_U", "pct_C"]
        # Ensure columns exist and are numeric
        valid_cols = [c for c in corr_cols if c in val_df.columns]

        correlations = val_df[valid_cols].corrwith(val_df["error"])
        print("Correlations with Error Magnitude:")
        print(correlations)
    else:
        print(
            f"Warning: Mismatch in validation samples. Preds: {len(rmse_per_sample)}, DF: {len(val_df)}"
        )

    # ------------------------------------------------------------------
    # 8. Submission Generation
    # ------------------------------------------------------------------
    THRESHOLD = 0.7247761841173526

    if final_val_loss < THRESHOLD:
        print(
            f"\nValidation metric {final_val_loss} < {THRESHOLD}. Generating submission..."
        )
        generate_submission(
            model_path=Config.MODEL_SAVE_PATH,
            output_path="./submission/submission.csv",
            device=device,
        )
    else:
        print(
            f"\nValidation metric {final_val_loss} >= {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
