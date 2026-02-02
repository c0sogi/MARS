import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_loader
from library.model import SR_DCN
from library.train import train_epoch, validate


def run_failure_analysis(model, val_loader, device, scored_indices):
    """
    Performs failure analysis by correlating sample-wise errors with metadata features.
    """
    model.eval()

    all_ids = []
    all_errors = []

    # Load metadata for correlation
    val_csv_path = os.path.join(Config.METADATA_DIR, "val.csv")
    if not os.path.exists(val_csv_path):
        print("Validation metadata not found, skipping failure analysis details.")
        return

    val_df = pd.read_csv(val_csv_path)
    # Ensure we can map IDs to metadata
    val_meta = val_df.set_index("id")

    print("\nRunning Failure Analysis on Validation Set...")

    with torch.no_grad():
        for x, y, p_idx, mask, ids in val_loader:
            x = x.to(device)
            y = y.to(device)
            p_idx = p_idx.to(device)
            mask = mask.to(device)

            # --- Pass 1: Cold Start ---
            b, l, _ = x.shape
            recycling_zero = torch.zeros((b, l, 5), device=device)
            pred1 = model(x, recycling_zero, p_idx)

            # --- Pass 2: Refinement ---
            recycling_input = pred1
            pred2 = model(x, recycling_input, p_idx)

            # Calculate Error Per Sample
            # preds: (B, L, 5), targets: (B, L, 5)
            # We calculate MCRMSE for each sample individually

            preds_scored = pred2[:, :, scored_indices]
            targets_scored = y[:, :, scored_indices]

            # Squared Error: (B, L, 3)
            se = (preds_scored - targets_scored) ** 2

            # Masking
            # mask is (B, L). Expand to (B, L, 3)
            mask_expanded = mask.unsqueeze(-1).expand_as(se)

            # Zero out masked positions
            se = se * mask_expanded

            # Sum over length: (B, 3)
            # Count valid positions per sample: (B, 1)
            valid_counts = mask.sum(dim=1).unsqueeze(-1)
            # Avoid div by zero
            valid_counts = torch.clamp(valid_counts, min=1.0)

            mse_per_col = se.sum(dim=1) / valid_counts  # (B, 3)
            rmse_per_col = torch.sqrt(mse_per_col + 1e-8)  # (B, 3)

            # Mean over columns -> MCRMSE per sample
            mcrmse_per_sample = rmse_per_col.mean(dim=1).cpu().numpy()

            all_ids.extend(ids)
            all_errors.extend(mcrmse_per_sample)

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame({"id": all_ids, "error": all_errors})

    # Merge with metadata
    merged_df = analysis_df.merge(val_meta, on="id", how="left")

    # Calculate Correlations
    # 1. Signal to Noise
    if "signal_to_noise" in merged_df.columns:
        corr, _ = stats.pearsonr(
            merged_df["error"], merged_df["signal_to_noise"].fillna(0)
        )
        print(f"Correlation (Error vs Signal_to_Noise): {corr:.4f}")

    # 2. Mean Reactivity (if available or calculated)
    if "mean_reactivity" in merged_df.columns:
        corr, _ = stats.pearsonr(
            merged_df["error"], merged_df["mean_reactivity"].fillna(0)
        )
        print(f"Correlation (Error vs Mean Reactivity): {corr:.4f}")

    # 3. Sequence Length (though constant 107, good check)
    if "seq_length" in merged_df.columns:
        # If std dev is 0, correlation is undefined/warning
        if merged_df["seq_length"].std() > 0:
            corr, _ = stats.pearsonr(merged_df["error"], merged_df["seq_length"])
            print(f"Correlation (Error vs Seq Length): {corr:.4f}")
        else:
            print("Correlation (Error vs Seq Length): N/A (Constant Length)")


def generate_submission(model, test_loader, device):
    """
    Generates submission file using the 2-pass strategy.
    """
    print("Generating submission for Test Set...")
    model.eval()
    results = []
    target_cols = Config.TARGET_COLS

    with torch.no_grad():
        for x, y, p_idx, mask, ids in test_loader:
            x = x.to(device)
            p_idx = p_idx.to(device)

            # --- Pass 1 ---
            b, l, _ = x.shape
            recycling_zero = torch.zeros((b, l, 5), device=device)
            pred1 = model(x, recycling_zero, p_idx)

            # --- Pass 2 ---
            recycling_input = pred1
            pred2 = model(x, recycling_input, p_idx)

            preds_np = pred2.cpu().numpy()

            for i, sample_id in enumerate(ids):
                sample_preds = preds_np[i]
                for seqpos in range(Config.SEQ_LENGTH):
                    row_id = f"{sample_id}_{seqpos}"
                    vals = sample_preds[seqpos]

                    row_dict = {"id_seqpos": row_id}
                    for k, col_name in enumerate(target_cols):
                        row_dict[col_name] = float(vals[k])
                    results.append(row_dict)

    submission_df = pd.DataFrame(results)
    cols_order = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols_order]

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Setup
    # Override Config for Fast Baseline
    Config.EPOCHS = 10
    Config.PATIENCE = 3

    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Scored indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    # 2. Data Loading
    print("Loading Data...")
    train_loader = get_loader("train", shuffle=True)
    val_loader = get_loader("val", shuffle=False)

    # 3. Model Initialization
    model = SR_DCN().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device, scored_indices)
        val_loss = validate(model, val_loader, device, scored_indices)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_loss:.6f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Final Evaluation & Failure Analysis
    print("\nLoading best model for final evaluation...")
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Recalculate metric on full validation set to be precise
    final_metric = validate(model, val_loader, device, scored_indices)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    run_failure_analysis(model, val_loader, device, scored_indices)

    # 6. Submission Logic
    THRESHOLD = 0.5417620723771521
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        test_loader = get_loader("test", shuffle=False)
        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
