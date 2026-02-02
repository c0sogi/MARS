import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, KLDivLossWithLogits, kl_divergence_score
from library.data import get_dataloaders
from library.model import (
    ChronologicallyEmbeddedDualStream,
    train_one_epoch,
    validate,
    predict,
)


def analyze_failures(val_df, y_true, y_pred):
    """
    Performs failure analysis by calculating per-sample KL divergence
    and correlating it with metadata features.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate KL per sample: sum(P * log(P/Q))
    # term1 = P * log(P)
    term1 = y_true * np.log(y_true + epsilon)
    # term2 = P * log(Q)
    term2 = y_true * np.log(y_pred)

    # KL = sum(term1 - term2)
    kl_per_sample = np.sum(term1 - term2, axis=1)

    print("\n==== Failure Analysis ====")
    print(f"Mean KL Error: {np.mean(kl_per_sample):.6f}")
    print(f"Max KL Error: {np.max(kl_per_sample):.6f}")

    # Correlate with features
    features = ["eeg_label_offset_seconds", "spectrogram_label_offset_seconds"]

    for feat in features:
        if feat in val_df.columns:
            # Handle potential NaNs in metadata (though unlikely based on EDA)
            valid_mask = ~val_df[feat].isna()
            if valid_mask.sum() > 1:
                corr, _ = pearsonr(
                    val_df.loc[valid_mask, feat], kl_per_sample[valid_mask]
                )
                print(f"Correlation between Error and {feat}: {corr:.4f}")
            else:
                print(f"Not enough valid data for {feat}")
        else:
            print(f"Feature {feat} not found in validation metadata.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Override Config for Fast Baseline
    # We use 3 epochs to ensure completion within 2 hours while allowing convergence
    epochs = 3
    batch_size = Config.BATCH_SIZE
    learning_rate = Config.LEARNING_RATE

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
        debug=False,  # Use full dataset for valid baseline
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = ChronologicallyEmbeddedDualStream(Config).to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    criterion = KLDivLossWithLogits()

    # 5. Training Loop
    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_kl = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val KL: {val_kl:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print("Saved Best Model.")

    # 6. Final Evaluation & Failure Analysis
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Generate predictions on Validation set for analysis
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            eeg, spec, rel, targets = batch
            eeg, spec, rel = eeg.to(device), spec.to(device), rel.to(device)

            logits = model(eeg, spec, rel)
            probs = F.softmax(logits, dim=1)

            val_probs.append(probs.cpu().numpy())
            val_targets.append(targets.numpy())

    val_probs = np.concatenate(val_probs, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Compute Final Metric
    final_metric = kl_divergence_score(val_targets, val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # Load Validation Metadata for Analysis
    val_df = pd.read_csv(Config.VAL_CSV)

    # Run Failure Analysis
    analyze_failures(val_df, val_targets, val_probs)

    # 7. Conditional Submission
    THRESHOLD = 0.7327804565429688

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        test_probs = predict(model, test_loader, device)

        # Load test metadata for IDs
        test_df = pd.read_csv(Config.TEST_CSV)

        submission = pd.DataFrame()
        submission["eeg_id"] = test_df["eeg_id"]
        for i, col in enumerate(Config.CLASS_NAMES):
            submission[col] = test_probs[:, i]

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
