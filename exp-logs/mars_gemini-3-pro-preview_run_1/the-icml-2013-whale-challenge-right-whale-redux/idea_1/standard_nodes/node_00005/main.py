import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import sys
import os

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, save_submission
from library.dataset import get_datasets
from library.model import BiGRUClassifier
from library.trainer import train_one_epoch, validate


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis by correlating error magnitude with input signal statistics.
    """
    model.eval()
    all_targets = []
    all_probs = []

    # Features to correlate
    feat_means = []
    feat_stds = []
    feat_maxs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy())

            # Extract features from the input spectrograms (Batch, 1, F, T)
            # We compute stats per sample in the batch
            # inputs shape: (B, 1, F, T) -> flatten last two dims for stats -> (B, F*T)
            flat_inputs = inputs.view(inputs.size(0), -1)

            feat_means.extend(flat_inputs.mean(dim=1).cpu().numpy())
            feat_stds.extend(flat_inputs.std(dim=1).cpu().numpy())
            feat_maxs.extend(flat_inputs.max(dim=1).values.cpu().numpy())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_probs)

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "spec_mean": feat_means,
            "spec_std": feat_stds,
            "spec_max": feat_maxs,
        }
    )

    print("\n--- Failure Analysis (Correlation with Error Magnitude) ---")
    correlations = df_analysis.corr()["error"].drop("error")
    print(correlations)
    print("---------------------------------------------------------")


def main():
    # 1. Configuration Overrides
    # Increase epochs to allow better convergence with augmentation
    Config.EPOCHS = 30

    # 2. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 3. Data Loading
    # Using cached data if available for speed
    train_dataset, val_dataset, test_dataset = get_datasets(load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    # 4. Model Initialization
    model = BiGRUClassifier().to(device)

    # Loss with class weighting
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # 5. Training Loop
    best_auc = 0.0

    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step(val_auc)

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

    # 6. Final Validation Metric
    # Load best model
    if os.path.exists(Config.MODEL_CHECKPOINT):
        model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))

    final_loss, final_auc = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 8. Submission Generation
    # Only generate submission if performance improves over the baseline
    BASELINE_AUC = 0.9788887700315186

    if final_auc > BASELINE_AUC:
        print(
            f"Validation AUC ({final_auc:.6f}) exceeds baseline ({BASELINE_AUC:.6f}). Generating submission..."
        )
        model.eval()
        predictions = []
        test_ids = []

        with torch.no_grad():
            for inputs, clip_ids in test_loader:
                inputs = inputs.to(device)

                logits = model(inputs)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                predictions.extend(probs)
                test_ids.extend(clip_ids)

        save_submission(predictions, test_ids, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation AUC ({final_auc:.6f}) did not exceed baseline ({BASELINE_AUC:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
