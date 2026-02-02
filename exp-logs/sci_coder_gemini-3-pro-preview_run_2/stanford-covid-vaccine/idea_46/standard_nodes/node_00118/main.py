import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.data import process_data, RNADataset
from library.model import SDFRNModel
from library.train import train_one_epoch, validate
from torch.utils.data import DataLoader

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_gc_content(sequence):
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def calculate_sample_errors(model, loader, device):
    """
    Computes MCRMSE for each sample in the loader.
    Returns a list of errors and a list of IDs.
    """
    model.eval()
    all_errors = []
    all_ids = []

    # Scored columns indices: reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_cols_indices = Config.TARGET_INDICES

    with torch.no_grad():
        # Iterate over loader. Note: DataLoader shuffles=False for Val/Test usually.
        # We need to be careful to align with metadata.
        # The loader yields batches.

        # To ensure alignment, we assume the loader is sequential (shuffle=False).
        # We will track the batch size.

        for batch in loader:
            seq = batch["seq"].to(device)
            struct = batch["struct"].to(device)
            loop = batch["loop"].to(device)
            pid = batch["partner_id"].to(device)
            pidx = batch["partner_idx"].to(device)
            targets = batch["targets"].to(device)

            # Pass 1
            pred1 = model(seq, struct, loop, pid, pidx, prev_pred=None)
            # Pass 2
            pred2 = model(seq, struct, loop, pid, pidx, prev_pred=pred1)

            # Calculate error per sample
            # pred2: (B, L, 5)
            # targets: (B, L, 5)

            preds_scored = pred2[:, : Config.PRED_LEN, scored_cols_indices]
            targets_scored = targets[:, : Config.PRED_LEN, scored_cols_indices]

            # MSE: (B, 68, 3)
            mse = (preds_scored - targets_scored) ** 2

            # RMSE per column per sample: (B, 3)
            rmse_per_col = torch.sqrt(mse.mean(dim=1))

            # MCRMSE per sample: (B, )
            mcrmse_per_sample = rmse_per_col.mean(dim=1)

            all_errors.extend(mcrmse_per_sample.cpu().numpy().tolist())

    return np.array(all_errors)


def main():
    # 1. Configuration
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override Config for Fast Baseline
    Config.EPOCHS = 15  # Reduced from 50 to ensure speed

    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_data = process_data("train", load_cached_data=True)
    val_data = process_data("val", load_cached_data=True)
    test_data = process_data("test", load_cached_data=True)

    train_dataset = RNADataset(train_data, "train")
    val_dataset = RNADataset(val_data, "val")
    test_dataset = RNADataset(test_data, "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Shuffle=False for validation to align with metadata for failure analysis
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = SDFRNModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            break

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Final Validation Metric
    final_val_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Load metadata to get features
    val_meta_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))

    # Calculate errors per sample
    sample_errors = calculate_sample_errors(model, val_loader, device)

    # Ensure alignment
    if len(sample_errors) != len(val_meta_df):
        print("Warning: Mismatch in validation set size for failure analysis.")
    else:
        # Extract features
        val_meta_df["error"] = sample_errors
        val_meta_df["gc_content"] = val_meta_df["sequence"].apply(get_gc_content)

        # Calculate correlations
        features = ["signal_to_noise", "SN_filter", "gc_content"]
        print("Correlation between Error and Features:")
        for feat in features:
            if feat in val_meta_df.columns:
                corr, _ = pearsonr(val_meta_df["error"], val_meta_df[feat])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not found in metadata")

    # 6. Submission
    THRESHOLD = 0.47142532743789534

    if final_val_metric < THRESHOLD:
        print("\nValidation metric meets threshold. Generating submission...")
        model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                seq = batch["seq"].to(device)
                struct = batch["struct"].to(device)
                loop = batch["loop"].to(device)
                pid = batch["partner_id"].to(device)
                pidx = batch["partner_idx"].to(device)

                # Pass 1
                pred1 = model(seq, struct, loop, pid, pidx, prev_pred=None)
                # Pass 2
                pred2 = model(seq, struct, loop, pid, pidx, prev_pred=pred1)

                all_preds.append(pred2.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)  # (N, 107, 5)
        test_ids = test_data["ids"]

        submission_rows = []
        for i, sample_id in enumerate(test_ids):
            sample_pred = all_preds[i]
            for j in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{j}"
                vals = sample_pred[j]
                submission_rows.append([row_id] + vals.tolist())

        sub_df = pd.DataFrame(
            submission_rows, columns=["id_seqpos"] + Config.ALL_TARGETS
        )
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {final_val_metric} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
