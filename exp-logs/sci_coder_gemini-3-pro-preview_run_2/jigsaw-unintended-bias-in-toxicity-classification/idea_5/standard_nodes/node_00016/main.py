import os
import sys
import torch
import numpy as np
import pandas as pd
import itertools
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_optimizer_params
from library.dataset import get_dataloaders
from library.model import ToxicityModel
from library.engine import Engine


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Load full datasets. Tokenization will be cached in ./working/idea_5
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Model...")
    model = ToxicityModel()
    model.to(device)

    # 4. Optimization Setup
    # Fast Baseline Strategy: Limit training steps to ensure completion within 2 hours.
    # RoBERTa-Large is computationally expensive. We process a subset of batches per epoch.
    MAX_STEPS_PER_EPOCH = 1500
    TOTAL_STEPS = MAX_STEPS_PER_EPOCH * Config.EPOCHS

    optimizer_params = get_optimizer_params(
        model,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        llrd_decay=Config.LLRD_DECAY,
    )

    optimizer = AdamW(optimizer_params, lr=Config.LEARNING_RATE)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=TOTAL_STEPS,
        pct_start=Config.WARMUP_RATIO,
    )

    engine = Engine(model, optimizer, scheduler, device)

    # 5. Training Loop
    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")

        # Create a limited iterator for this epoch to speed up training
        limited_train_loader = itertools.islice(train_loader, MAX_STEPS_PER_EPOCH)

        # Train on the limited subset
        engine.train_one_epoch(limited_train_loader, epoch + 1)

    # 6. Validation
    print("Starting Validation...")
    # engine.validate computes the specific competition metric (Weighted ROC-AUCs)
    val_loss, val_score = engine.validate(val_loader)

    # Required Output Format
    print(f"Final Validation Metric: {val_score}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # We need raw predictions and targets to calculate correlations.
    # We perform a dedicated inference pass on the validation set.
    model.eval()

    all_preds_fa = []
    all_targets_fa = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            # Device-side trimming (reusing logic from Engine)
            max_len = int(attention_mask.sum(dim=1).max().item())
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]

            # Forward pass (only toxicity logits needed for error analysis)
            toxicity_logits, _ = model(input_ids, attention_mask)

            preds = torch.sigmoid(toxicity_logits).cpu().numpy().flatten()
            target_vals = targets.cpu().numpy()

            all_preds_fa.append(preds)
            all_targets_fa.append(target_vals)

    # Concatenate results
    all_preds_fa = np.concatenate(all_preds_fa)
    all_targets_fa = np.concatenate(all_targets_fa, axis=0)

    # Calculate Error Magnitude
    # Target is at index 0
    main_targets = all_targets_fa[:, 0]
    errors = np.abs(all_preds_fa - main_targets)

    # Construct DataFrame for correlation analysis
    fa_df = pd.DataFrame({"error": errors, "target": main_targets})

    # Add identity columns (Indices 1 to N in targets)
    for idx, col_name in enumerate(Config.IDENTITY_COLUMNS):
        fa_df[col_name] = all_targets_fa[:, idx + 1]

    # Calculate and print correlations
    print("Correlation between Error Magnitude and Input Features:")
    correlations = fa_df.corr()["error"].sort_values(ascending=False)
    # Remove self-correlation
    correlations = correlations.drop("error", errors="ignore")
    print(correlations)

    # 8. Submission Generation
    THRESHOLD = 0.9273793163893314

    if val_score > THRESHOLD:
        print(
            f"\nValidation Score ({val_score}) exceeds threshold ({THRESHOLD}). Generating Submission..."
        )

        all_test_preds = []
        all_test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                ids = batch["ids"]  # Keep IDs on CPU

                # Device-side trimming
                max_len = int(attention_mask.sum(dim=1).max().item())
                input_ids = input_ids[:, :max_len]
                attention_mask = attention_mask[:, :max_len]

                # Forward pass
                toxicity_logits, _ = model(input_ids, attention_mask)
                preds = torch.sigmoid(toxicity_logits).cpu().numpy().flatten()

                all_test_preds.append(preds)
                all_test_ids.append(ids.numpy())

        # Concatenate
        all_test_preds = np.concatenate(all_test_preds)
        all_test_ids = np.concatenate(all_test_ids)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"id": all_test_ids, "prediction": all_test_preds})

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Score ({val_score}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
