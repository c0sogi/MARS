import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from library.config import Config
from library.utils import seed_everything, mcrmse
from library.dataset import get_dataloaders
from library.model import TopologicalWideResBiLSTM
from library.train import train_fn, inference_fn


def evaluate_and_collect(model, loader, device, config):
    """
    Evaluates the model and collects IDs, predictions, and targets for analysis.
    """
    model.eval()
    all_ids = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["id"]
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            distance = batch["distance"].to(device)
            rwpe = batch["rwpe"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            preds = model(sequence, loop_type, rwpe, distance)

            # Slice to scored length (68)
            preds_sliced = preds[:, : config.pred_len, :]

            all_ids.extend(ids)
            all_preds.append(preds_sliced.cpu())
            all_targets.append(targets.cpu())

    return all_ids, torch.cat(all_preds), torch.cat(all_targets)


def run_failure_analysis(ids, preds, targets, metadata_path):
    """
    Correlates model error with input features.
    """
    print("\n=== Failure Analysis ===")

    # 1. Calculate per-sample RMSE (Error Magnitude)
    # preds, targets shape: (N, 68, 3)
    # MSE per sample: mean over (68, 3) -> (N,)
    mse_per_sample = torch.mean((preds - targets) ** 2, dim=(1, 2)).numpy()
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 2. Load Metadata to get features
    if not os.path.exists(metadata_path):
        print(
            f"Metadata file not found at {metadata_path}. Skipping detailed feature analysis."
        )
        return

    df_meta = pd.read_parquet(metadata_path)
    # Filter/Order metadata to match the order of ids in validation set
    # Create a mapping
    id_to_idx = {id_: i for i, id_ in enumerate(ids)}

    # We need to ensure we align the dataframe rows with our error arrays
    # Filter df to only include validation ids
    df_val = df_meta[df_meta["id"].isin(ids)].copy()

    # Reindex df_val to match the order of 'ids' list
    df_val["sort_idx"] = df_val["id"].map(id_to_idx)
    df_val = df_val.sort_values("sort_idx").reset_index(drop=True)

    # 3. Construct Features
    # Signal to Noise
    if "signal_to_noise" in df_val.columns:
        sn_ratio = df_val["signal_to_noise"].values
    else:
        sn_ratio = np.zeros(len(df_val))

    # Sequence Composition
    seqs = df_val["sequence"].values
    len_A = np.array([s.count("A") for s in seqs])
    len_G = np.array([s.count("G") for s in seqs])
    len_C = np.array([s.count("C") for s in seqs])
    len_U = np.array([s.count("U") for s in seqs])

    # Structure Composition
    structs = df_val["structure"].values
    paired = np.array(
        [s.count("(") for s in structs]
    )  # Count open brackets (half of total paired)

    features = {
        "Signal_to_Noise": sn_ratio,
        "Count_A": len_A,
        "Count_G": len_G,
        "Count_C": len_C,
        "Count_U": len_U,
        "Paired_Bases": paired,
    }

    # 4. Compute Correlations
    print(f"Correlations with Error Magnitude (RMSE) [N={len(rmse_per_sample)}]:")
    for name, feat_values in features.items():
        if len(feat_values) != len(rmse_per_sample):
            continue
        # Check for constant values
        if np.std(feat_values) == 0:
            print(f"  {name}: Constant value (cannot correlate)")
            continue

        corr, pval = spearmanr(rmse_per_sample, feat_values)
        print(f"  {name}: Spearman R = {corr:.4f} (p={pval:.4g})")


def main():
    # 1. Configuration
    # Using 15 epochs for a fast but effective baseline
    config = Config(epochs=15, batch_size=16)
    seed_everything(config.seed)

    if config.device == "cuda":
        torch.cuda.empty_cache()

    print(f"Running on device: {config.device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = TopologicalWideResBiLSTM(config)
    model.to(config.device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 4. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(config.working_dir, "best_model.pth")

    print("Starting training...")
    for epoch in range(config.epochs):
        # Train
        train_loss = train_fn(
            model, train_loader, optimizer, criterion, config.device, config
        )

        # Validation (using simple metric check first)
        # We reuse the evaluate_and_collect but just for metric here to save time?
        # Actually eval_fn in library is faster as it doesn't store IDs.
        # But we need IDs for final analysis. We'll use library eval_fn for the loop.
        from library.train import eval_fn

        val_score = eval_fn(model, val_loader, config.device, config)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    print("Training complete.")

    # 5. Final Evaluation & Failure Analysis
    print("Loading best model for analysis...")
    model.load_state_dict(torch.load(best_model_path, map_location=config.device))

    ids, preds, targets = evaluate_and_collect(model, val_loader, config.device, config)

    # Compute Final Metric
    final_metric = mcrmse(targets, preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    run_failure_analysis(ids, preds, targets, config.val_file)

    # 6. Conditional Submission
    THRESHOLD = 0.6176461577
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference on Test Set
        test_ids, test_preds = inference_fn(model, test_loader, config.device)

        # Format Submission
        submission_rows = []
        # test_preds shape: (N, 107, 3)

        for i, sample_id in enumerate(test_ids):
            sample_preds = test_preds[i]

            for seqpos in range(config.seq_len):
                row_id = f"{sample_id}_{seqpos}"

                # Extract predictions
                reactivity = sample_preds[seqpos, 0]
                deg_Mg_pH10 = sample_preds[seqpos, 1]
                deg_Mg_50C = sample_preds[seqpos, 2]

                # Fill unscored columns
                deg_pH10 = 0.0
                deg_50C = 0.0

                submission_rows.append(
                    [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
                )

        columns = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        submission_df = pd.DataFrame(submission_rows, columns=columns)

        submission_df.to_csv(config.submission_file, index=False)
        print(f"Submission saved to {config.submission_file}")
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
