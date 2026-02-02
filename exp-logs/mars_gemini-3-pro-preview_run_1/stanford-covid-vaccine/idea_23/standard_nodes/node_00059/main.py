import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, mcrmse
from library.dataset import get_dataloaders
from library.model import RNAModel
from library.engine import Engine, SWAHandler


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis by correlating prediction errors with input features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    # 1. Collect per-sample errors
    all_ids = []
    all_errors = []

    criterion = torch.nn.MSELoss(reduction="none")

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            # Forward pass
            outputs = model(seq, loop, dist)

            # Slice to scored length (68)
            pred_scored = outputs[:, : Config.PRED_LEN, :]
            target_scored = targets[:, : Config.PRED_LEN, :]

            # Compute MSE per sample (average over seq_len and channels)
            # Shape: (B, 68, 3) -> (B,)
            loss = criterion(pred_scored, target_scored)
            sample_mse = loss.mean(dim=(1, 2)).cpu().numpy()

            all_ids.extend(ids)
            all_errors.extend(sample_mse)

    # 2. Load Metadata to get features
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)

    # Create a dataframe for errors
    df_errors = pd.DataFrame({"id": all_ids, "error": all_errors})

    # Merge with metadata
    df_analysis = pd.merge(df_errors, df_val, on="id", how="inner")

    # 3. Extract features for correlation
    # Signal to Noise
    if "signal_to_noise" in df_analysis.columns:
        snr = df_analysis["signal_to_noise"].fillna(0).values
        corr, _ = pearsonr(df_analysis["error"], snr)
        print(f"Correlation (Error vs Signal-to-Noise): {corr:.4f}")

    # SN Filter
    if "SN_filter" in df_analysis.columns:
        sn_filter = df_analysis["SN_filter"].astype(int).values
        corr, _ = pearsonr(df_analysis["error"], sn_filter)
        print(f"Correlation (Error vs SN_filter): {corr:.4f}")

    # Sequence Composition (A, G, C, U content)
    # We calculate this from the sequence string
    df_analysis["pct_A"] = df_analysis["sequence"].apply(
        lambda x: x.count("A") / len(x)
    )
    df_analysis["pct_G"] = df_analysis["sequence"].apply(
        lambda x: x.count("G") / len(x)
    )
    df_analysis["pct_U"] = df_analysis["sequence"].apply(
        lambda x: x.count("U") / len(x)
    )

    for base in ["A", "G", "U"]:
        corr, _ = pearsonr(df_analysis["error"], df_analysis[f"pct_{base}"])
        print(f"Correlation (Error vs %{base}): {corr:.4f}")


def generate_submission(engine, test_loader):
    """
    Generates the submission file.
    """
    print("\nGenerating submission...")
    ids, preds = engine.predict(test_loader)

    # preds shape: (N, 107, 3)
    # Target columns in model output: ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    submission_data = []

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # Shape (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"

            # Extract predictions
            reactivity = float(sample_preds[seqpos, 0])
            deg_Mg_pH10 = float(sample_preds[seqpos, 1])
            deg_Mg_50C = float(sample_preds[seqpos, 2])

            # Unscored columns set to 0.0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_data.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    submission_df = pd.DataFrame(
        submission_data,
        columns=[
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ],
    )

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Setup
    _ = Config()  # Initialize config (creates directories)
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Device: {device}")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = RNAModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    engine = Engine(model, device, optimizer, scheduler)
    swa_handler = SWAHandler(Config.SWA_START_EPOCH, Config.WORKING_DIR)

    # 4. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = engine.train_one_epoch(train_loader, epoch)
        val_loss, val_score = engine.evaluate(val_loader)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Collect SWA checkpoint
        swa_handler.update(model, epoch)

        # Silent progress (only required info printed at end usually, but minimal logs help debug)
        # print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | MCRMSE: {val_score:.4f}")

    # 5. SWA Finalization
    print("Finalizing SWA Model...")
    avg_state = swa_handler.finalize()

    if avg_state is not None:
        model.load_state_dict(avg_state)
        # Save SWA model
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    else:
        print("SWA failed, using last model state.")
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 6. Final Evaluation
    _, final_metric = engine.evaluate(val_loader)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 8. Conditional Submission
    THRESHOLD = 0.6209375959946717

    if final_metric < THRESHOLD:
        generate_submission(engine, test_loader)
    else:
        print(f"Validation metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
