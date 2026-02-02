import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, save_checkpoint, kl_divergence
from library.dataset import EEGSeizureDataset
from library.models import CyclicFusionNet
from library.engine import train_one_epoch, validate, inference


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Create necessary directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.OUTPUT_SUBMISSION_PATH), exist_ok=True)

    # 2. Data Loading
    train_df_full = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Validation Loader (Static)
    val_dataset = EEGSeizureDataset(val_df, mode="val", augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = CyclicFusionNet()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate steps for OneCycleLR based on average fold size
    avg_fold_size = len(train_df_full) / Config.NUM_FOLDS
    steps_per_epoch = int(np.ceil(avg_fold_size / Config.BATCH_SIZE))

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.TOTAL_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # 4. Cyclic Training Loop
    # Create disjoint patient folds
    unique_patients = train_df_full["patient_id"].unique()
    rng = np.random.default_rng(Config.SEED)
    rng.shuffle(unique_patients)
    patient_folds = np.array_split(unique_patients, Config.NUM_FOLDS)

    best_score = float("inf")

    # We will limit the run to ensure it fits within time constraints if needed,
    # but the cyclic nature (subsets) is already efficient.

    for epoch in range(Config.TOTAL_EPOCHS):
        # Determine current fold
        fold_idx = epoch % Config.NUM_FOLDS

        # Create subset for this epoch
        current_patients = patient_folds[fold_idx]
        train_subset = train_df_full[
            train_df_full["patient_id"].isin(current_patients)
        ].copy()

        train_dataset = EEGSeizureDataset(train_subset, mode="train", augment=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        # Train
        train_loss = train_one_epoch(
            train_loader, model, optimizer, scheduler, device, epoch
        )

        # Validate
        val_loss = validate(val_loader, model, device)

        # Save Best Model
        if val_loss < best_score:
            best_score = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_score, Config.MODEL_SAVE_PATH
            )

    # 5. Final Validation & Metric Calculation
    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    # Get predictions on validation set for metric and failure analysis
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for eeg, spec, targets in val_loader:
            eeg = eeg.to(device)
            spec = spec.to(device)
            outputs = model(eeg, spec)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Compute Final Metric
    # Using the utility logic but applied to the whole array
    # Clip predictions for stability
    epsilon = 1e-15
    val_preds_clipped = np.clip(val_preds, epsilon, 1.0 - epsilon)

    # KL Divergence: sum(P * log(P/Q)) where P=True, Q=Pred?
    # Note: The task metric description says "Kullback Liebler divergence between the predicted probability and the observed target."
    # Standard KL(P || Q) is usually sum(P * log(P/Q)).
    # PyTorch F.kl_div(log_pred, target) computes sum(target * (log(target) - log_pred)) which is KL(target || pred).
    # The utility function `kl_divergence` uses F.kl_div(torch.log(y_pred), y_true, reduction="batchmean").
    # This matches the competition standard (Target || Prediction).

    # We use the provided utility function logic manually for the full set to be precise
    val_preds_tensor = torch.tensor(val_preds_clipped)
    val_targets_tensor = torch.tensor(val_targets)

    final_metric = F.kl_div(
        torch.log(val_preds_tensor), val_targets_tensor, reduction="batchmean"
    ).item()

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    # Calculate per-sample loss
    # KL is calculated as sum(target * (log(target) - log(pred))) per sample
    # Note: log(target) might be -inf if target is 0, but x*log(x) -> 0 as x->0.
    # We compute it carefully.

    log_preds = np.log(val_preds_clipped)
    # Handle target log safely
    log_targets = np.zeros_like(val_targets)
    mask = val_targets > 0
    log_targets[mask] = np.log(val_targets[mask])

    # KL per sample (sum over classes)
    kl_per_sample = np.sum(val_targets * (log_targets - log_preds), axis=1)

    # Create analysis dataframe
    analysis_df = val_df.copy()
    analysis_df["error"] = kl_per_sample

    print("\n=== Failure Analysis ===")
    print("Correlation between Error and Metadata features:")

    features_to_check = ["eeg_label_offset_seconds", "spectrogram_label_offset_seconds"]

    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Handle NaNs just in case
            valid_mask = analysis_df[feat].notna()
            if valid_mask.sum() > 1:
                corr, _ = pearsonr(
                    analysis_df.loc[valid_mask, feat],
                    analysis_df.loc[valid_mask, "error"],
                )
                print(f"{feat}: {corr:.4f}")

    # Check correlation with target classes (are some classes harder?)
    print("\nCorrelation between Error and Target Probabilities:")
    for col in Config.TARGET_COLS:
        if col in analysis_df.columns:  # raw votes
            pass  # skip raw votes, look at probs if available or calculate

    # Calculate prob cols if not in df (metadata generation script added them as *_prob)
    prob_cols = [c.replace("_vote", "_prob") for c in Config.TARGET_COLS]

    for p_col in prob_cols:
        if p_col in analysis_df.columns:
            corr, _ = pearsonr(analysis_df[p_col], analysis_df["error"])
            print(f"{p_col}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.6822116374969482

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = EEGSeizureDataset(test_df, mode="test", augment=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds = inference(test_loader, model, device)

        sub_df = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
        sub_df["eeg_id"] = test_df["eeg_id"]

        # Ensure column order
        cols = ["eeg_id"] + Config.TARGET_COLS
        sub_df = sub_df[cols]

        # Normalize
        vote_cols = Config.TARGET_COLS
        sub_df[vote_cols] = sub_df[vote_cols].div(sub_df[vote_cols].sum(axis=1), axis=0)

        sub_df.to_csv(Config.OUTPUT_SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.OUTPUT_SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
