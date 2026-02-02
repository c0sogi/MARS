import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.optim as optim

# Import from library
from library.config import Config
from library.utils import set_seed, MCRMSELoss, compute_val_metric
from library.data import get_dataloaders
from library.model import RNAModel
from library.engine import train_fn, eval_fn, predict_fn, generate_submission


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for fast baseline execution
    Config.EPOCHS = 15
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 2

    # Setup environment
    set_seed(Config.SEED)
    Config.setup_directories()
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing Model...")
    model = RNAModel(Config).to(device)

    # ==========================================
    # 4. Training
    # ==========================================
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = MCRMSELoss()

    best_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model_runfile.pth")

    print("Starting Training Loop...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)
        val_score = eval_fn(model, val_loader, device)
        scheduler.step()

        # Simple logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.5f}"
        )

        # Checkpoint
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val Score: {best_score:.5f}")

    # ==========================================
    # 5. Final Evaluation
    # ==========================================
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Compute final metric
    final_val_score = eval_fn(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_score}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("\nRunning Failure Analysis...")
    # Load validation metadata to get features
    val_df = pd.read_parquet(Config.VAL_PARQUET)

    # Get predictions for validation set
    val_preds_dict = predict_fn(model, val_loader, device)

    # Indices for scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS
    scored_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    ids = []
    errors = []

    for _, row in val_df.iterrows():
        sample_id = row["id"]
        if sample_id not in val_preds_dict:
            continue

        # Get prediction: (107, 5) -> slice to (68, 3)
        pred_full = val_preds_dict[sample_id]
        pred_scored = pred_full[:68, scored_indices]

        # Get target: list -> array (68, 3)
        targets_list = []
        for col in scored_cols:
            # Ensure we take the first 68 elements
            t_val = row[col]
            if isinstance(t_val, (list, np.ndarray)):
                targets_list.append(t_val[:68])
            else:
                # Fallback if somehow scalar (unlikely given parquet)
                targets_list.append([t_val] * 68)

        target_matrix = np.array(targets_list).T  # (68, 3)

        # Compute RMSE for this sample
        mse = np.mean((pred_scored - target_matrix) ** 2)
        rmse = np.sqrt(mse)

        ids.append(sample_id)
        errors.append(rmse)

    analysis_df = pd.DataFrame({"id": ids, "error": errors})

    # Merge with metadata features
    # Features of interest: signal_to_noise, SN_filter
    # Also calculate nucleotide content
    meta_cols = ["id", "signal_to_noise", "SN_filter", "sequence"]
    merged_df = analysis_df.merge(val_df[meta_cols], on="id")

    # Feature Engineering
    merged_df["A_pct"] = merged_df["sequence"].apply(lambda x: x.count("A") / len(x))
    merged_df["G_pct"] = merged_df["sequence"].apply(lambda x: x.count("G") / len(x))
    merged_df["C_pct"] = merged_df["sequence"].apply(lambda x: x.count("C") / len(x))
    merged_df["U_pct"] = merged_df["sequence"].apply(lambda x: x.count("U") / len(x))

    # Calculate Correlations
    print("Correlation between Error and Features:")
    features = ["signal_to_noise", "SN_filter", "A_pct", "G_pct", "C_pct", "U_pct"]
    for feat in features:
        if feat in merged_df.columns:
            corr = merged_df["error"].corr(merged_df[feat])
            print(f"  {feat}: {corr:.6f}")

    # ==========================================
    # 7. Submission
    # ==========================================
    THRESHOLD = 0.5884495377540588
    if final_val_score < THRESHOLD:
        print(
            f"\nValidation score ({final_val_score}) < Threshold ({THRESHOLD}). Generating submission..."
        )
        test_preds = predict_fn(model, test_loader, device)
        generate_submission(test_preds, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation score ({final_val_score}) >= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
