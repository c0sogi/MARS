import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_device, ensure_directory
from library.dataset import load_datasets
from library.model_arch import ToxicityModel
from library.engine import Engine
from library.swa_utils import SWAHandler
from library.metrics import BiasMetricCalculator


def main():
    # 1. Setup
    print("Initializing run...")
    seed_everything(Config.SEED)
    device = get_device()
    ensure_directory(os.path.dirname(Config.SUBMISSION_PATH))

    # 2. Load Data
    print("Loading datasets...")
    # Force load_cached_data=True as per instructions
    train_dataset, val_dataset, test_dataset = load_datasets(load_cached_data=True)

    # FAST BASELINE: Subset training data to ensure completion within 2 hours
    # RoBERTa-Large is computationally expensive. 100k samples is a safe upper bound for a quick check.
    MAX_TRAIN_SAMPLES = 100000
    if len(train_dataset) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsetting training data from {len(train_dataset)} to {MAX_TRAIN_SAMPLES} for fast baseline."
        )
        indices = np.arange(MAX_TRAIN_SAMPLES)
        train_dataset = Subset(train_dataset, indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = ToxicityModel(Config.MODEL_NAME)
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate total steps for OneCycleLR
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * Config.EPOCHS

    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # SWA Handler
    swa_handler = SWAHandler(
        model, swa_start_epoch=Config.SWA_START_EPOCH, swa_lr=Config.SWA_LR
    )

    # 5. Training Loop
    engine = Engine(model, optimizer, device, scheduler, swa_handler)

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        train_loss = engine.train_epoch(train_loader, epoch)

        # Optional: Quick validation check (metrics calculation is heavy, so maybe skip or do on subset)
        # For this baseline, we will trust the final validation.
        print(f"Epoch {epoch + 1} completed. Loss: {train_loss:.4f}")

    # 6. SWA Finalization
    print("\nApplying Stochastic Weight Averaging...")
    swa_handler.swap_swa_params(model)

    # 7. Final Validation
    print("Running final validation on full validation set...")
    val_loss, val_metrics = engine.evaluate(val_loader)

    final_score = val_metrics["final_score"]
    print(f"Final Validation Metric: {final_score}")

    # 8. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Re-run a quick pass to get raw data for correlation analysis since evaluate aggregates internally
    # Alternatively, we can extract data from the dataset directly since we need targets and aux_targets
    # But we need the model's specific predictions.

    # Let's collect predictions manually to link with aux_targets
    model.eval()
    all_preds = []
    all_targets = []
    all_aux = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            aux_targets = batch["aux_targets"].to(device, non_blocking=True)

            # Trim
            input_ids, attention_mask = engine._trim_batch(input_ids, attention_mask)

            tox_logits, _ = model(input_ids, attention_mask)
            preds = torch.sigmoid(tox_logits).squeeze(-1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_aux.append(aux_targets.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    aux_data = np.concatenate(all_aux)

    # Calculate Error
    errors = np.abs(y_pred - y_true)

    # Correlation with Identities
    print("Correlation between Absolute Error and Identity Attributes:")
    identity_cols = Config.IDENTITY_COLS

    # Create a DataFrame for easy correlation
    analysis_df = pd.DataFrame(aux_data, columns=identity_cols)
    analysis_df["error"] = errors

    correlations = analysis_df.corr()["error"].drop("error")
    print(correlations.sort_values(ascending=False))

    # 9. Submission
    THRESHOLD = 0.9273793163893314
    if final_score > THRESHOLD:
        print(
            f"\nValidation score ({final_score:.6f}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.VALID_BATCH_SIZE,  # Use valid batch size for inference
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        predictions_dict = engine.predict(test_loader)

        # Create submission DataFrame
        submission = pd.DataFrame.from_dict(
            predictions_dict, orient="index", columns=["prediction"]
        )
        submission.index.name = "id"
        submission.reset_index(inplace=True)

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation score ({final_score:.6f}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
