import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score

from library.config import Config
from library.utils import set_seed, ModelEMA, map_fine_to_coarse
from library.dataset import get_dataloaders
from library.transforms import AudioTransforms
from library.model import get_model


def train_epoch(model, ema, transforms, loader, criterion, optimizer, device):
    """
    Performs one epoch of training with Mixup and EMA updates.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for waveforms, labels, _ in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # 1. GPU Augmentation & Feature Extraction
        # Returns: mixed_features, labels_a, labels_b, lam
        features, targets_a, targets_b, lam = transforms(
            waveforms, labels, train=True, mixup_alpha=Config.MIXUP_ALPHA
        )

        # 2. Forward Pass
        optimizer.zero_grad()
        outputs = model(features)

        # 3. Mixup Loss Calculation
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
            outputs, targets_b
        )

        # 4. Backward & Step
        loss.backward()
        optimizer.step()

        # 5. Update EMA Model
        ema.update(model)

        # Statistics
        running_loss += loss.item() * waveforms.size(0)

        # Accuracy (Approximation using the dominant label in mixup)
        # If lam > 0.5, target_a is dominant, else target_b
        _, predicted = torch.max(outputs.data, 1)
        dominant_labels = torch.where(lam > 0.5, targets_a, targets_b)
        total += labels.size(0)
        correct += (predicted == dominant_labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, transforms, loader, criterion, device):
    """
    Validates the model (typically the EMA model) on the validation set.
    Computes both Fine-Grained Accuracy (for model selection) and
    Competition Metric (for reference).
    """
    model.eval()
    running_loss = 0.0

    # Lists to store predictions for metric calculation
    all_preds_fine = []
    all_labels_fine = []

    with torch.no_grad():
        for waveforms, labels, _ in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            # 1. Feature Extraction (No Augmentation)
            features = transforms(waveforms, labels=None, train=False)

            # 2. Forward Pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * waveforms.size(0)

            # 3. Collect Predictions
            _, predicted = torch.max(outputs, 1)
            all_preds_fine.extend(predicted.cpu().numpy())
            all_labels_fine.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    # Fine-Grained Accuracy (36 classes)
    fine_acc = accuracy_score(all_labels_fine, all_preds_fine)

    # Competition Metric Accuracy (12 classes)
    # Map indices to string labels, then map fine-grained strings to target strings
    pred_labels_str = [Config.get_label_from_index(idx) for idx in all_preds_fine]
    true_labels_str = [Config.get_label_from_index(idx) for idx in all_labels_fine]

    mapped_preds = map_fine_to_coarse(pred_labels_str)
    mapped_true = map_fine_to_coarse(true_labels_str)

    comp_acc = accuracy_score(mapped_true, mapped_preds)

    return avg_loss, fine_acc, comp_acc


def run_training(epochs=Config.EPOCHS, load_cached_data=True):
    """
    Main execution function for training.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Preparation
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Model & Components Initialization
    print("Initializing Model and Transforms...")
    model = get_model(device)

    # EMA Wrapper
    ema = ModelEMA(model, decay=0.999, device=device)

    # GPU Transforms
    transforms = AudioTransforms(device=device)

    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 3. Training Loop
    best_fine_acc = 0.0
    patience = 10
    patience_counter = 0
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, ema, transforms, train_loader, criterion, optimizer, device
        )

        # Validate (using EMA model)
        val_loss, val_fine_acc, val_comp_acc = validate(
            ema.ema_model, transforms, val_loader, criterion, device
        )

        # Step Scheduler
        scheduler.step()

        # Logging
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss:.6f} | Train Acc (Approx): {train_acc:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val Fine-Grained Acc: {val_fine_acc:.10f}")
        print(f"  Val Competition Acc:  {val_comp_acc:.10f}")

        # Model Selection (Based on Fine-Grained Accuracy)
        if val_fine_acc > best_fine_acc:
            best_fine_acc = val_fine_acc
            patience_counter = 0
            torch.save(ema.ema_model.state_dict(), save_path)
            print(f"  -> New Best Model Saved! (Acc: {best_fine_acc:.6f})")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(
        f"Training complete. Best Fine-Grained Validation Accuracy: {best_fine_acc:.10f}"
    )
    return save_path
