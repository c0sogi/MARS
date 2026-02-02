import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from library.config import Config
from library.utils import set_seed
from library.loss import MaskedMCRMSELoss
from library.data import get_loader
from library.model import InteractionEnrichedDenseNet
from library.train import train_epoch, validate

# Constants
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
TARGET_METRIC_THRESHOLD = 0.5417620723771521
BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set by correlating
    sample-wise error with metadata features.
    """
    print("\nPerforming Failure Analysis...")

    # Load validation metadata to get features like signal_to_noise
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure model is in eval mode
    model.eval()

    sample_rmses = []
    scored_indices = Config.SCORED_TARGET_INDICES

    with torch.no_grad():
        # Iterate over validation loader (shuffle=False is guaranteed by get_loader('val'))
        for inputs, partner_indices, targets in val_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.cpu().numpy()

            # Forward pass
            preds = model(inputs, partner_indices).cpu().numpy()

            # Align predictions to target length if necessary
            if preds.shape[1] > targets.shape[1]:
                preds = preds[:, : targets.shape[1], :]

            # Filter for scored columns only
            preds_scored = preds[:, :, scored_indices]
            targets_scored = targets[:, :, scored_indices]

            # Calculate RMSE per sample
            # (Batch, Seq, Cols) -> Mean over Seq and Cols -> Sqrt
            squared_diff = (preds_scored - targets_scored) ** 2
            mse_per_sample = np.mean(squared_diff, axis=(1, 2))
            rmse_per_sample = np.sqrt(mse_per_sample)

            sample_rmses.extend(rmse_per_sample)

    # Add error metric to dataframe
    # Note: val_loader preserves order of val.csv
    val_df["rmse_error"] = sample_rmses

    # Calculate correlations
    features_to_check = [
        "signal_to_noise",
        "mean_reactivity",
        "SN_filter",
        "seq_length",
    ]
    print("Correlation between Sample RMSE and Metadata Features:")
    for feat in features_to_check:
        if feat in val_df.columns:
            corr = val_df["rmse_error"].corr(val_df[feat])
            print(f"  {feat}: {corr:.4f}")


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves to ./submission/submission.csv
    """
    print("\nGenerating submission file...")

    # Load test data
    test_loader = get_loader(
        "test", batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )
    ids = test_loader.dataset.ids

    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, partner_indices, _ in test_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Output shape: (Batch, SeqLen, 5)
            outputs = model(inputs, partner_indices)
            all_preds.append(outputs.cpu().numpy())

    # Concatenate all batches: (N_samples, SeqLen, 5)
    preds_array = np.concatenate(all_preds, axis=0)

    submission_data = []

    # Format for submission
    for idx, sample_id in enumerate(ids):
        sample_preds = preds_array[idx]

        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            vals = sample_preds[seqpos]

            # Order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            row_dict = {
                "id_seqpos": row_id,
                "reactivity": vals[0],
                "deg_Mg_pH10": vals[1],
                "deg_pH10": vals[2],
                "deg_Mg_50C": vals[3],
                "deg_50C": vals[4],
            }
            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Ensure directory exists
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR)

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using full dataset for optimal performance
    train_loader = get_loader(
        "train", batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )
    val_loader = get_loader(
        "val", batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # 3. Model Initialization
    model = InteractionEnrichedDenseNet().to(device)

    # 4. Optimization
    criterion = MaskedMCRMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # 5. Training Loop
    best_mcrmse = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, device)
        val_mcrmse = val_metrics["mcrmse"]

        # Update Scheduler
        scheduler.step(val_mcrmse)

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_mcrmse:.5f}"
        )

    # 6. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    final_metrics = validate(model, val_loader, device)
    final_score = final_metrics["mcrmse"]

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_score}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Conditional Submission
    if final_score < TARGET_METRIC_THRESHOLD:
        generate_submission(model, device)
    else:
        print(
            f"\nValidation metric {final_score} is not lower than threshold {TARGET_METRIC_THRESHOLD}."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
