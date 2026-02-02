import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import copy
import os
import sys

# Import from provided libraries
from library.config import (
    Config,
    DeepParallelVectorDCNResNet,
    VectorDCNLayer,
    ResNetBlock,
)
from library.data_loader import get_dataloaders

# =============================================================================
# Model Architecture (Aliases to match Target File Description)
# =============================================================================

# The architecture is already implemented in library.config matching the Idea.
# We alias them here to fulfill the specific class naming requirements of model.py.


class VectorCrossLayer(VectorDCNLayer):
    """
    Alias for VectorDCNLayer: Rank-1 Cross Layer with near-zero initialization.
    """

    pass


class PreActSwishResNetBlock(ResNetBlock):
    """
    Alias for ResNetBlock: Full Pre-Activation Block with Swish.
    """

    pass


class HybridSwishModel(DeepParallelVectorDCNResNet):
    """
    Alias for DeepParallelVectorDCNResNet:
    Parallel Vector-DCN and Swish-ResNet backbone with Non-Linear Projection Head.
    """

    pass


def get_model(input_dim, num_classes):
    """
    Factory function to instantiate the model.
    """
    # Config.HIDDEN_DIM, Config.DCN_LAYERS etc. are used internally by the class
    model = HybridSwishModel(input_dim, num_classes)
    return model


# =============================================================================
# Training & Inference Pipeline
# =============================================================================


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def predict(model, dataloader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            preds.append(predicted.cpu().numpy())

    return np.concatenate(preds)


def main():
    print("Starting Deep Parallel Vector-DCN-ResNet (Swish Variant) Pipeline...")

    # 1. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, num_features, num_classes, test_ids = (
        get_dataloaders(load_cached_data=True)
    )

    print(f"Features: {num_features}, Classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 2. Model Initialization
    device = Config.DEVICE
    print(f"Using device: {device}")

    model = get_model(num_features, num_classes).to(device)

    # 3. Optimizer & Scheduler
    # Idea: AdamW (Decoupled Weight Decay)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Idea: ReduceLROnPlateau with aggressive decay (0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    early_stopping_patience = 10  # Reasonable patience for 60 epochs

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} Acc: {val_acc:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save best model to disk
            torch.save(
                model.state_dict(), os.path.join(Config.WORKING_DIR, "best_model.pth")
            )
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Accuracy: {best_acc:.6f}")

    # 5. Inference
    print("Loading best model for inference...")
    model.load_state_dict(best_model_wts)

    print("Generating predictions on test set...")
    predictions = predict(model, test_loader, device)

    # Inverse transform labels if necessary
    # The data_loader returns a label_encoder in the meta dict, but get_dataloaders doesn't return it directly.
    # However, process_data caches the label_encoder.
    # We need to map 0..N-1 back to original class IDs (1, 2, 3, 4, 6, 7).

    # Load metadata to get label encoder
    meta_path = os.path.join(Config.CACHE_DIR, "metadata.npy")
    if os.path.exists(meta_path):
        meta_dict = np.load(meta_path, allow_pickle=True).item()
        le = meta_dict["label_encoder"]
        final_preds = le.inverse_transform(predictions)
    else:
        # Fallback if cache missing (unlikely given get_dataloaders passed)
        # Assuming classes are [1, 2, 3, 4, 6, 7] sorted
        # But safer to fail or assume standard mapping if strictly needed.
        # Given the pipeline, the cache must exist.
        print(
            "Warning: Metadata cache not found for label inverse transform. Using raw predictions."
        )
        final_preds = predictions

    # 6. Submission
    submission = pd.DataFrame({"Id": test_ids, "Cover_Type": final_preds})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# Execute the pipeline
if __name__ == "__main__":
    main()
