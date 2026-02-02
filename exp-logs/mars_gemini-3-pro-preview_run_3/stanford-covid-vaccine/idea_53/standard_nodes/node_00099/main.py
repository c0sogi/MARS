import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import warnings

# Import library modules
from library.config import Config
from library.dataset import RNADataset
from library.model import DeepStabilizedBiGRU
from library.engine import train_fn, eval_fn

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance against metadata features.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # 1. Collect Predictions and Targets
    with torch.no_grad():
        for batch in val_loader:
            sequence = batch["sequence"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            outputs = model(sequence, pair_indices)

            # Slice to scored length for error calculation
            outputs_sliced = outputs[:, : Config.PRED_LEN, :]

            all_preds.append(outputs_sliced.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 2. Calculate Per-Sample Error (MCRMSE on scored columns)
    # Scored indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = Config.SCORED_INDICES

    preds_scored = all_preds[:, :, scored_indices]
    targets_scored = all_targets[:, :, scored_indices]

    # MSE per sample: (N, 68, 3) -> mean over (68, 3) -> (N,)
    # Note: MCRMSE definition is root-mean-squared, usually averaged over columns.
    # Here we calculate a scalar error metric per sample to correlate.
    mse_per_sample = torch.mean((preds_scored - targets_scored) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # 3. Load Metadata to correlate
    # We need to match IDs to metadata features
    val_df = pd.read_parquet(Config.VAL_PARQUET)
    val_meta = val_df.set_index("id")

    # Create analysis dataframe
    analysis_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

    # Merge with metadata
    analysis_df = analysis_df.merge(
        val_meta[["signal_to_noise", "SN_filter", "sequence"]], on="id", how="left"
    )

    # Calculate sequence features
    analysis_df["pct_A"] = analysis_df["sequence"].apply(
        lambda s: s.count("A") / len(s)
    )
    analysis_df["pct_U"] = analysis_df["sequence"].apply(
        lambda s: s.count("U") / len(s)
    )
    analysis_df["pct_G"] = analysis_df["sequence"].apply(
        lambda s: s.count("G") / len(s)
    )
    analysis_df["pct_C"] = analysis_df["sequence"].apply(
        lambda s: s.count("C") / len(s)
    )

    # 4. Calculate Correlations
    correlations = {}
    features = ["signal_to_noise", "SN_filter", "pct_A", "pct_U", "pct_G", "pct_C"]

    print("Correlation between Error and Features:")
    for feat in features:
        if feat in analysis_df.columns:
            corr = analysis_df["error"].corr(analysis_df[feat])
            correlations[feat] = corr
            print(f"  {feat}: {corr:.4f}")

    return correlations


def generate_submission(model, test_loader, device):
    """
    Generates submission file for the test set.
    """
    print("\n==== Generating Submission ====")
    model.eval()

    ids_seqpos = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            sequence = batch["sequence"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            ids = batch["id"]

            # Forward pass (N, 107, 5)
            outputs = model(sequence, pair_indices)
            outputs = outputs.cpu().numpy()

            # Prepare data for submission format
            for i, sample_id in enumerate(ids):
                # For each position in sequence (length 107)
                for seq_idx in range(Config.SEQ_LEN):
                    ids_seqpos.append(f"{sample_id}_{seq_idx}")
                    preds_list.append(outputs[i, seq_idx, :])

    # Create DataFrame
    cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    preds_array = np.array(preds_list)

    sub_df = pd.DataFrame(preds_array, columns=cols)
    sub_df.insert(0, "id_seqpos", ids_seqpos)

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {sub_df.shape}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    train_dataset = RNADataset(split="train", debug=Config.DEBUG)
    val_dataset = RNADataset(split="val", debug=Config.DEBUG)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = DeepStabilizedBiGRU().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    best_score = float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_fn(model, train_loader, optimizer, device, scheduler)

        # Evaluate
        val_score = eval_fn(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.5f}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  >>> New Best Model Saved! (Score: {best_score:.5f})")

    # 5. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    final_val_score = eval_fn(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_score}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 7. Submission
    THRESHOLD = 0.5884495377540588

    if final_val_score < THRESHOLD:
        print(
            f"\nValidation score ({final_val_score:.5f}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = RNADataset(split="test", debug=Config.DEBUG)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nValidation score ({final_val_score:.5f}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
