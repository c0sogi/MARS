import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import soundfile as sf
from library import config, utils, model, dataset, train


def perform_failure_analysis(val_metadata_path, targets, probs):
    """
    Analyzes the correlation between prediction errors and input signal characteristics.
    """
    print("\n--- Failure Analysis ---")

    # Calculate errors
    targets = np.array(targets)
    probs = np.array(probs)
    errors = np.abs(targets - probs)

    # Load metadata to access file paths
    df = pd.read_csv(val_metadata_path)

    # Ensure lengths match
    if len(df) != len(errors):
        print(
            f"Warning: Metadata length ({len(df)}) matches predictions ({len(errors)}) mismatch. Skipping detailed analysis."
        )
        return

    # Extract features
    rms_values = []
    peak_values = []
    durations = []

    print(f"Extracting features for {len(df)} validation files...")
    # Limit to a subset if too large to save time, but 4500 is manageable
    for idx, row in df.iterrows():
        file_path = os.path.join(config.INPUT_DIR, row["file_path"])
        try:
            # Read audio
            wav, sr = sf.read(file_path)

            # Handle multi-channel
            if wav.ndim > 1:
                wav = np.mean(wav, axis=1)

            rms = np.sqrt(np.mean(wav**2))
            peak = np.max(np.abs(wav))
            duration = len(wav) / sr

            rms_values.append(rms)
            peak_values.append(peak)
            durations.append(duration)
        except Exception as e:
            # Fill with mean or skip
            rms_values.append(0)
            peak_values.append(0)
            durations.append(0)

    # Compute correlations
    features = {
        "RMS Energy": rms_values,
        "Peak Amplitude": peak_values,
        "Duration": durations,
    }

    print("Correlation between Error Magnitude and Features:")
    for name, values in features.items():
        if len(values) == len(errors):
            corr = np.corrcoef(values, errors)[0, 1]
            print(f"{name}: {corr:.4f}")
        else:
            print(f"{name}: Size mismatch")


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    # Using cached data for speed
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    print(f"Initializing model: {config.MODEL_NAME}")
    net = model.WhaleEfficientNet(pretrained=config.PRETRAINED)
    net = net.to(device)

    # 4. Optimization Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN
    )

    # 5. Training Loop
    best_score = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    # Limit epochs for fast baseline if needed, but config.EPOCHS=20 is fine for A100
    epochs = config.EPOCHS

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train.train_one_epoch(
            net, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_auc = train.validate(net, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.6f}"
        )

        # Checkpointing
        if val_auc > best_score:
            best_score = val_auc
            utils.save_checkpoint(net, optimizer, epoch, best_score, best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Final Evaluation & Failure Analysis
    print("\nTraining complete. Loading best model for final evaluation...")

    # Load best model
    best_net = model.WhaleEfficientNet(pretrained=False)
    best_net = best_net.to(device)
    _, loaded_score = utils.load_checkpoint(best_net, best_model_path, device=device)

    # Final Validation Pass to get predictions for analysis
    best_net.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float().view(-1, 1)
            outputs = best_net(inputs)
            probs = torch.sigmoid(outputs)

            all_targets.extend(targets.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    final_auc = utils.calculate_roc_auc(all_targets, all_probs)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    perform_failure_analysis(config.VAL_METADATA, all_targets, all_probs)

    # 7. Conditional Submission
    THRESHOLD = 0.994260809807678

    if final_auc > THRESHOLD:
        print(
            f"\nMetric ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        train.generate_submission(best_net, test_loader, device, config.SUBMISSION_PATH)
    else:
        print(f"\nMetric ({final_auc}) <= Threshold ({THRESHOLD}). Submission skipped.")


if __name__ == "__main__":
    main()
