import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchaudio
from library.config import Config
from library.dataset import get_dataloaders
from library.model import TimePreservingEfficientNet
from library.engine import train_one_epoch, evaluate, predict
from library.utils import set_seed


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Computes correlation between error magnitude and audio features.
    """
    print("Starting Failure Analysis...")
    model.eval()

    # 1. Get Predictions and Targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            output = model(data)
            preds = torch.sigmoid(output).cpu().numpy().flatten()
            targets = target.numpy().flatten()

            all_preds.extend(preds)
            all_targets.extend(targets)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Error Magnitude
    errors = np.abs(all_targets - all_preds)

    # 2. Extract Features from Validation Files
    # We need to read the validation CSV to get filepaths matching the loader order
    val_df = pd.read_csv(Config.VAL_CSV)
    if Config.DEBUG:
        val_df = val_df.iloc[:100]

    features = {"rms": [], "max_amp": [], "zcr": []}

    print(f"Extracting audio features for {len(val_df)} validation samples...")

    for idx, row in val_df.iterrows():
        filepath = os.path.join(Config.INPUT_ROOT, row["filepath"])
        try:
            # Load audio
            waveform, sr = torchaudio.load(filepath)
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = waveform.squeeze()

            # RMS Energy
            rms = torch.sqrt(torch.mean(waveform**2)).item()
            features["rms"].append(rms)

            # Max Amplitude
            max_amp = torch.max(torch.abs(waveform)).item()
            features["max_amp"].append(max_amp)

            # Zero Crossing Rate
            if len(waveform) > 1:
                zcr = ((waveform[:-1] * waveform[1:]) < 0).float().mean().item()
            else:
                zcr = 0.0
            features["zcr"].append(zcr)

        except Exception:
            # Fallback for read errors
            features["rms"].append(0.0)
            features["max_amp"].append(0.0)
            features["zcr"].append(0.0)

    # 3. Compute Correlations
    print("\nFailure Analysis - Correlation with Error Magnitude:")
    for feat_name, feat_values in features.items():
        if len(feat_values) != len(errors):
            continue

        # Use numpy for correlation
        if np.std(feat_values) > 1e-9 and np.std(errors) > 1e-9:
            corr = np.corrcoef(feat_values, errors)[0, 1]
            print(f"Feature: {feat_name:20s} | Correlation: {corr:.4f}")
        else:
            print(f"Feature: {feat_name:20s} | Correlation: 0.0000 (Low variance)")
    print("-" * 30)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    Config.setup()

    # Override Config for efficient baseline execution
    # 20 epochs is sufficient for convergence with pre-trained EfficientNet
    Config.EPOCHS = 20

    Config.print_config()

    # 2. Data Loading
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = TimePreservingEfficientNet()
    model = model.to(device)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
        min_lr=Config.MIN_LR,
    )

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_auc)

        # Logging
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc:.6f} | LR: {lr:.2e}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best Model Saved! AUC: {best_auc:.6f}")

    # 6. Final Evaluation
    print("\nLoading Best Model for Final Evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Re-evaluate on full validation set to get the exact metric
    _, final_auc = evaluate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_auc:.16f}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 8. Conditional Submission
    THRESHOLD = 0.9934990421176494

    if final_auc > THRESHOLD:
        print(
            f"Validation metric {final_auc} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        predict(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation metric {final_auc} does not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
