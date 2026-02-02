import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time

from library.config import (
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    SEED,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    TEST_CSV,
    WORKING_DIR,
    NUM_WORKERS,
)
from library.utils import (
    set_seed,
    calculate_accuracy,
    LabelEncoder,
    save_checkpoint,
    load_checkpoint,
)
from library.transforms import AudioProcessor
from library.dataset import get_dataloaders
from library.modules import MultiResSKCRNN


def train_one_epoch(model, processor, dataloader, criterion, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    processor.train()  # Enables noise injection and SpecAugment

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # 1. GPU-Accelerated Augmentation & Feature Extraction
        # waveforms: (B, T), specs: (B, 3, F, T)
        waveforms_aug, specs_aug = processor(inputs)

        # 2. Forward Pass
        outputs = model(waveforms_aug, specs_aug)

        # 3. Loss & Backprop
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)

        with torch.no_grad():
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == targets.data)
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    return epoch_loss, epoch_acc.item()


def evaluate(model, processor, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    processor.eval()  # Disables noise injection and SpecAugment

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Feature Extraction (No Augmentation)
            waveforms, specs = processor(inputs)

            # Forward Pass
            outputs = model(waveforms, specs)

            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == targets.data)
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    return epoch_loss, epoch_acc.item()


def train_pipeline(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    load_cached_data=True,
    patience=5,
):
    """
    Main training pipeline with Early Stopping.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 2. Model & Processor Initialization
    print("Initializing Model and Processor...")
    model = MultiResSKCRNN().to(device)
    processor = AudioProcessor().to(device)

    # 3. Setup Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 4. Training Loop
    best_val_acc = 0.0
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, processor, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, processor, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{epochs} | Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Checkpoint & Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, MODEL_SAVE_PATH)
            print(f"  -> New best model saved! (Acc: {best_val_acc:.6f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.6f}")
    return best_val_acc


def generate_submission(
    model_path=MODEL_SAVE_PATH,
    output_path=SUBMISSION_PATH,
    load_cached_data=True,
):
    """
    Generates predictions for the test set and saves to CSV.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating submission using model at {model_path}...")

    # 1. Load Data
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 2. Load Model
    model = MultiResSKCRNN().to(device)
    processor = AudioProcessor().to(device)

    # Load weights
    try:
        load_checkpoint(model_path, model, device=device)
    except FileNotFoundError:
        print("Error: Best model checkpoint not found. Cannot generate submission.")
        return

    model.eval()
    processor.eval()

    # 3. Inference
    all_preds = []
    label_encoder = LabelEncoder()

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            # Process
            waveforms, specs = processor(inputs)
            outputs = model(waveforms, specs)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    # 4. Map to Labels
    pred_labels = label_encoder.decode_batch(all_preds)

    # 5. Match with Filenames
    # We read the test metadata to get the file paths in order
    df_test = pd.read_csv(TEST_CSV)

    # Extract filename from filepath (e.g., "test/audio/clip_001.wav" -> "clip_001.wav")
    fnames = df_test["filepath"].apply(os.path.basename).tolist()

    if len(fnames) != len(pred_labels):
        raise ValueError(
            f"Mismatch: {len(fnames)} files vs {len(pred_labels)} predictions."
        )

    # 6. Save Submission
    submission_df = pd.DataFrame({"fname": fnames, "label": pred_labels})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(submission_df.head())
