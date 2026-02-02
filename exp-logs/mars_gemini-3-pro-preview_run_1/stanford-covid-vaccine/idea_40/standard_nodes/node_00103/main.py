import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, mcrmse
from library.data import get_dataloaders
from library.model import WideResBiGRU
from library.loss import MaskedMSELoss
from library.train import train_one_epoch, validate, generate_submission


def main():
    # 1. Configuration for Fast Baseline
    # Reverting to 20 epochs for better convergence (Cite {solution_lesson_node_00056})
    Config.EPOCHS = 20
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # Using cached data for speed
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=Config.DEBUG
    )

    # 3. Model Setup
    model = WideResBiGRU().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = MaskedMSELoss(scoring_length=Config.PRED_LEN)

    # 4. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    for epoch in range(Config.EPOCHS):
        # Train
        _ = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Save Best Model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Evaluation
    # Load best model for analysis and submission
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Collect validation predictions for full metric calculation and failure analysis
    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop_type"].to(device)
            dist = batch["pair_dist"].to(device)
            tgt = batch["target"].to(device)
            ids = batch["id"]

            out = model(seq, loop, dist)

            # Slice to scored length for correct metric calculation
            out = out[:, : Config.PRED_LEN, :]
            tgt = tgt[:, : Config.PRED_LEN, :]

            val_preds.append(out.cpu().numpy())
            val_targets.append(tgt.cpu().numpy())
            val_ids.extend(ids)

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate and print final metric
    final_metric = mcrmse(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nFailure Analysis (Correlation with Error):")

    # Calculate RMSE per sample (scalar error metric)
    # Shape: (N_samples, 68, 3) -> (N_samples,)
    mse_per_sample = np.mean((val_targets - val_preds) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load metadata to get features
    if os.path.exists(Config.VAL_METADATA):
        df_val = pd.read_parquet(Config.VAL_METADATA)

        # Map errors to dataframe using ID to ensure alignment
        error_map = dict(zip(val_ids, rmse_per_sample))
        df_val["error"] = df_val["id"].map(error_map)

        # Derive sequence composition features
        df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
        df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
        df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
        df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))

        # Features to analyze
        analysis_cols = [
            "signal_to_noise",
            "SN_filter",
            "len_A",
            "len_G",
            "len_C",
            "len_U",
        ]

        for col in analysis_cols:
            if col in df_val.columns:
                # Drop NaNs just in case (though metadata should be clean)
                valid_data = df_val[[col, "error"]].dropna()
                if len(valid_data) > 0:
                    corr = valid_data[col].corr(valid_data["error"])
                    print(f"  {col}: {corr:.4f}")
    else:
        print("  Validation metadata not found, skipping detailed analysis.")

    # 7. Submission
    THRESHOLD = 0.6199890971183777
    if final_metric < THRESHOLD:
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
