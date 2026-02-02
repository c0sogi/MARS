import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import sys

# Import from the provided library files
from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import PCANet, MaskedMAELoss
from library.utils import clear_cache


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline execution
    Config.EPOCHS = 6
    Config.BATCH_SIZE = 128

    # Limit batches per epoch to ensure quick runtime (approx 640k samples/epoch)
    # Total dataset is ~4.3M samples, so this sees a diverse subset each epoch.
    MAX_BATCHES_PER_EPOCH = 5000

    # Submission Threshold
    THRESHOLD = 0.23978149890899658

    # Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print(f"Initializing Fast Baseline on {device}...")
    print(f"Config: {Config.EPOCHS} Epochs, Max {MAX_BATCHES_PER_EPOCH} batches/epoch")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # Use load_cached_data=True to utilize pre-processed data in ./working
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # ==========================================
    # 3. Model & Optimizer
    # ==========================================
    model = PCANet(Config).to(device)

    criterion = MaskedMAELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0
        train_batches = 0

        # Train on full dataset
        for i, batch in enumerate(train_loader):
            inputs = batch["input"].to(device)
            u_out = batch["u_out"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets, u_out)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_accum / train_batches if train_batches > 0 else 0.0

        # Validation (Full Set)
        model.eval()
        val_loss_accum = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                u_out = batch["u_out"].to(device)
                targets = batch["target"].to(device)

                preds = model(inputs)
                loss = criterion(preds, targets, u_out)

                val_loss_accum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_accum / val_batches if val_batches > 0 else 0.0

        # Scheduler Step
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train MAE: {avg_train_loss:.6f} | Val MAE: {avg_val_loss:.6f}"
        )

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # ==========================================
    # 5. Final Evaluation & Failure Analysis
    # ==========================================
    print("\nLoading best model for final evaluation...")
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    model.eval()

    val_loss_accum = 0
    val_count = 0

    # Containers for failure analysis
    all_errors = []
    all_inputs = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(device)
            u_out = batch["u_out"].to(device)
            targets = batch["target"].to(device)

            preds = model(inputs)

            # Metric Calculation
            loss = criterion(preds, targets, u_out)
            val_loss_accum += loss.item()
            val_count += 1

            # Failure Analysis Data Collection
            # We analyze errors only in the inspiratory phase (u_out == 0)
            # Using < 0.5 for float safety
            mask = u_out < 0.5

            # Flatten to sample level
            mask_flat = mask.view(-1).cpu().numpy().astype(bool)

            if mask_flat.sum() > 0:
                abs_error = torch.abs(preds - targets)
                error_flat = abs_error.view(-1).cpu().numpy()[mask_flat]

                # Flatten inputs: (Batch, Seq, Feat) -> (Batch*Seq, Feat)
                inputs_flat = inputs.view(-1, Config.INPUT_DIM).cpu().numpy()[mask_flat]

                all_errors.append(error_flat)
                all_inputs.append(inputs_flat)

    final_metric = val_loss_accum / val_count
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    if all_errors:
        print("\nPerforming Failure Analysis...")
        all_errors = np.concatenate(all_errors)
        all_inputs = np.concatenate(all_inputs, axis=0)

        analysis_df = pd.DataFrame(all_inputs, columns=Config.FEATURES)
        analysis_df["error"] = all_errors

        correlations = (
            analysis_df.corr()["error"].drop("error").sort_values(ascending=False)
        )
        print("Correlation between Error Magnitude and Input Features:")
        print(correlations)

    # ==========================================
    # 6. Submission
    # ==========================================
    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold {THRESHOLD}. Generating submission..."
        )

        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input"].to(device)
                preds = model(inputs)
                predictions.append(preds.cpu().numpy().flatten())

        all_preds = np.concatenate(predictions)

        # Load sample submission to ensure correct IDs and format
        sample_sub = pd.read_csv(
            os.path.join(Config.INPUT_DIR, "sample_submission.csv")
        )

        # Handle potential length mismatch (though pipeline should prevent this)
        if len(all_preds) != len(sample_sub):
            print(
                f"Warning: Prediction count {len(all_preds)} differs from submission file {len(sample_sub)}."
            )
            if len(all_preds) > len(sample_sub):
                all_preds = all_preds[: len(sample_sub)]

        sample_sub["pressure"] = all_preds
        sample_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
