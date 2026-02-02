import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import from provided library files
from library.utils import seed_everything, get_device, save_submission
from library.data_processing import load_and_preprocess_data
from library.model import ParallelDCNResNeXt


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

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


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())

    return np.array(all_preds)


def run_training(
    batch_size=4096,
    epochs=60,
    learning_rate=1e-3,
    patience=10,
    base_dir="./working/idea_11",
    sample_size=None,
):
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    seed_everything(42)
    device = get_device()
    os.makedirs(base_dir, exist_ok=True)
    model_save_path = os.path.join(base_dir, "parallel_dcn_resnext.pth")

    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = load_and_preprocess_data(
        batch_size=batch_size,
        load_cached_data=True,
        base_dir=base_dir,
        sample_size=sample_size,
    )

    # Determine input dimension from a batch
    sample_input, _ = next(iter(train_loader))
    input_dim = sample_input.shape[1]

    # Determine number of classes.
    # Classes are 1-7. Dataset subtracts 1, so internal labels are 0-6.
    # We need 7 output neurons.
    num_classes = 7

    print(f"Input Dimension: {input_dim}")
    print(f"Num Classes: {num_classes}")

    # 3. Model Initialization
    model = ParallelDCNResNeXt(
        input_dim=input_dim,
        num_classes=num_classes,
        dcn_layers=3,
        resnext_layers=3,
        d_model=1024,
        cardinality=32,
    ).to(device)

    # 4. Optimization Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3
    )

    # 5. Training Loop with Early Stopping
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.8f} | Train Acc: {train_acc:.8f} | "
            f"Val Loss: {val_loss:.8f} | Val Acc: {val_acc:.8f}"
        )

        # Early Stopping Check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            # Save checkpoint immediately to disk as well
            torch.save(best_model_state, model_save_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f} seconds.")
    print(f"Best Validation Accuracy: {best_val_acc:.8f}")

    # 6. Load Best Model for Inference
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        # Fallback if training failed completely (unlikely)
        torch.save(model.state_dict(), model_save_path)

    # 7. Generate Submission
    print("Generating predictions on test set...")
    raw_preds = predict(model, test_loader, device)

    # Map predictions back to original labels (0-6 -> 1-7)
    final_preds = raw_preds + 1

    submission_path = "./submission/submission.csv"
    save_submission(test_ids, final_preds, output_path=submission_path)

    return best_val_acc
