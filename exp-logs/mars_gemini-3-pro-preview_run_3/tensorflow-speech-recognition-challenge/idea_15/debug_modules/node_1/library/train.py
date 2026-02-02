import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import set_seed, MetricMonitor, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders, load_background_noises
from library.transforms import GPUAudioPreprocess
from library.model import FrequencyAttentiveResNeStCRNN


def train_one_epoch(
    epoch, model, preprocessor, loader, optimizer, criterion, device, noise_data=None
):
    """
    Trains the model for one epoch using the GPU-native pipeline.
    """
    model.train()
    preprocessor.train()  # Enable augmentations

    metric_monitor = MetricMonitor()

    # Iterate over raw waveforms
    for batch_idx, (waveforms, targets) in enumerate(loader):
        waveforms = waveforms.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 1. GPU-Native Preprocessing & Augmentation
        # noise_data is already on GPU if provided
        spectrograms = preprocessor(waveforms, training=True, noise=noise_data)

        # 2. Forward Pass
        optimizer.zero_grad()
        logits = model(spectrograms)

        # 3. Loss Calculation
        loss = criterion(logits, targets)

        # 4. Backward Pass
        loss.backward()
        optimizer.step()

        # 5. Metrics
        with torch.no_grad():
            # Calculate accuracy
            _, preds = torch.max(logits, 1)
            accuracy = (preds == targets).float().mean()

        metric_monitor.update("loss", loss.item(), n=waveforms.size(0))
        metric_monitor.update("accuracy", accuracy.item(), n=waveforms.size(0))

    return metric_monitor.get("loss"), metric_monitor.get("accuracy")


def validate(model, preprocessor, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    preprocessor.eval()  # Disable augmentations

    metric_monitor = MetricMonitor()

    with torch.no_grad():
        for batch_idx, (waveforms, targets) in enumerate(loader):
            waveforms = waveforms.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # GPU Preprocessing (No Augmentation)
            spectrograms = preprocessor(waveforms, training=False)

            # Forward Pass
            logits = model(spectrograms)
            loss = criterion(logits, targets)

            # Metrics
            _, preds = torch.max(logits, 1)
            accuracy = (preds == targets).float().mean()

            metric_monitor.update("loss", loss.item(), n=waveforms.size(0))
            metric_monitor.update("accuracy", accuracy.item(), n=waveforms.size(0))

    return metric_monitor.get("loss"), metric_monitor.get("accuracy")


def predict_and_submit(model, preprocessor, loader, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    preprocessor.eval()

    all_preds = []

    # Iterate test set
    with torch.no_grad():
        for waveforms, _ in loader:
            waveforms = waveforms.to(device)
            spectrograms = preprocessor(waveforms, training=False)
            logits = model(spectrograms)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())

    # Map IDs to Labels
    pred_labels = [Config.ID2LABEL[p] for p in all_preds]

    # Get filenames from the dataset metadata
    # The loader preserves order since shuffle=False
    test_metadata = loader.dataset.metadata

    # Extract just the filename (e.g., clip_00000.wav) from the full path
    fnames = test_metadata["filepath"].apply(lambda x: os.path.basename(x)).values

    # Create Submission DataFrame
    submission = pd.DataFrame({"fname": fnames, "label": pred_labels})

    # Save
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(debug=False, max_samples=None, epochs=None):
    """
    Main execution function for training and evaluation.
    """
    # 1. Setup
    config = Config(debug=debug, max_samples=max_samples, epochs=epochs)
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.display()

    # 2. Data Loading
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=config.DEBUG, max_samples=config.MAX_SAMPLES
    )

    # Load background noise for augmentation
    noise_data = load_background_noises(load_cached_data=True)
    if noise_data is not None:
        noise_data = noise_data.to(device)
        print("Background noise loaded to GPU.")
    else:
        print("Warning: Background noise not found.")

    # 3. Model & Components Initialization
    print("Initializing Model and Preprocessor...")
    model = FrequencyAttentiveResNeStCRNN().to(device)
    preprocessor = GPUAudioPreprocess(device=device).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    best_acc = 0.0
    patience = 5
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(1, config.EPOCHS + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            epoch,
            model,
            preprocessor,
            train_loader,
            optimizer,
            criterion,
            device,
            noise_data,
        )

        # Validate
        val_loss, val_acc = validate(model, preprocessor, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Print Metrics
        print(
            f"Epoch {epoch}/{config.EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_acc, filename="best_model.pth"
            )
            patience_counter = 0
            print(f"  -> New Best Model Saved! (Acc: {best_acc:.6f})")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    # 5. Final Inference
    print("\nGenerating Submission...")
    # Load best model
    _, best_score = load_checkpoint(model, filename="best_model.pth")
    print(f"Loaded best model with Validation Accuracy: {best_score:.6f}")

    predict_and_submit(model, preprocessor, test_loader, device, Config.SUBMISSION_PATH)

    return best_acc
