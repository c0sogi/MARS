import os
import sys
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
from library.config import Config
from library.utils import seed_everything, MCRMSE
from library.dataset import get_dataloader
from library.model import EIPFN
from library.train import train_model


def main():
    # 1. Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)

    # 2. Training
    # Run training for 15 epochs to ensure convergence within time limits
    print("Starting training...")
    train_model(debug=False, epochs=15, batch_size=32)

    # 3. Validation & Metric Calculation
    print("Starting validation inference...")
    val_loader = get_dataloader(
        mode="val", load_cached_data=True, shuffle=False, batch_size=64
    )

    # Load best model
    model = EIPFN().to(device)
    model_path = Config.MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Training might have failed.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    val_preds = []
    val_targets = []

    # Run Inference
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            p_idx = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            # Pass 1 (Zero Feedback)
            p1 = model(inputs, p_idx, y_prev=None)
            # Pass 2 (Feedback from Pass 1)
            p2 = model(inputs, p_idx, y_prev=p1)

            val_preds.append(p2.cpu().numpy())
            val_targets.append(targets.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Compute Global MCRMSE
    # Create mask for valid positions (first 68)
    mask = np.zeros((val_preds.shape[0], val_preds.shape[1]), dtype=bool)
    mask[:, : Config.SEQ_SCORED] = True

    metric_calc = MCRMSE(scored_indices=Config.TARGET_INDICES)
    metric_calc.update(val_preds, val_targets, mask)
    final_score = metric_calc.compute()

    print(f"Final Validation Metric: {final_score:.16f}")

    # 4. Failure Analysis
    print("Performing failure analysis...")
    # Load metadata
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    val_df = pd.read_csv(val_meta_path)

    # Calculate per-sample RMSE (on scored columns and valid positions)
    scored_cols = Config.TARGET_INDICES
    per_sample_errors = []

    for i in range(len(val_preds)):
        sample_sse = 0.0
        sample_count = 0
        for col in scored_cols:
            p = val_preds[i, : Config.SEQ_SCORED, col]
            t = val_targets[i, : Config.SEQ_SCORED, col]
            sample_sse += np.sum((p - t) ** 2)
            sample_count += len(p)

        rmse = np.sqrt(sample_sse / sample_count) if sample_count > 0 else 0.0
        per_sample_errors.append(rmse)

    val_df["error"] = per_sample_errors

    # Correlation: Signal to Noise
    if "signal_to_noise" in val_df.columns:
        corr_sn, _ = stats.pearsonr(val_df["signal_to_noise"], val_df["error"])
        print(f"Correlation (Signal-to-Noise vs Error): {corr_sn:.4f}")

    # Correlation: GC Content
    val_df["gc_content"] = val_df["sequence"].apply(
        lambda x: (x.count("G") + x.count("C")) / len(x)
    )
    corr_gc, _ = stats.pearsonr(val_df["gc_content"], val_df["error"])
    print(f"Correlation (GC Content vs Error): {corr_gc:.4f}")

    # 5. Submission
    threshold = 0.47142532743789534
    if final_score < threshold:
        print(
            f"Validation score meets threshold ({threshold}). Generating submission..."
        )

        test_loader = get_dataloader(
            mode="test", load_cached_data=True, shuffle=False, batch_size=64
        )
        test_meta_path = os.path.join(Config.METADATA_DIR, "test.csv")
        test_df = pd.read_csv(test_meta_path)
        test_ids = test_df["id"].values

        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(device)
                p_idx = batch["partner_indices"].to(device)

                # Pass 1
                p1 = model(inputs, p_idx, y_prev=None)
                # Pass 2
                p2 = model(inputs, p_idx, y_prev=p1)

                test_preds.append(p2.cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)  # (N, 107, 5)

        # Format Submission
        submission_rows = []
        # Columns in output: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        # Indices: 0, 1, 2, 3, 4

        for i, sample_id in enumerate(test_ids):
            pred_matrix = test_preds[i]  # (107, 5)
            for j in range(len(pred_matrix)):
                row_id = f"{sample_id}_{j}"
                vals = pred_matrix[j]
                submission_rows.append([row_id] + list(vals))

        cols = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        sub_df = pd.DataFrame(submission_rows, columns=cols)

        sub_path = "./submission/submission.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"Validation score {final_score} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
