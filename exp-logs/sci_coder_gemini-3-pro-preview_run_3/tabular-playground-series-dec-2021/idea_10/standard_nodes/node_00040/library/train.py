import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import copy
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import ParallelDCN_SE_ResNet


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
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

    return running_loss / total, correct / total


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
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / total, correct / total


def fit_model(epochs=60, batch_size=4096, quick_run=False):
    """
    Orchestrates the training process, including data loading, model initialization,
    training loop with early stopping, and submission generation.
    """
    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size,
        load_cached_data=True,
        cache_dir="./working/idea_10/",
        quick_run=quick_run,
    )

    # Determine input dimension from the first batch
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]
    num_classes = 7  # Cover Types 1-7 are mapped to 0-6 internally

    # 3. Model Initialization
    model = ParallelDCN_SE_ResNet(input_dim, num_classes).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 8
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} Acc: {train_acc} | Val Loss: {val_loss} Acc: {val_acc}"
        )

        # Scheduler Step
        scheduler.step(val_acc)

        # Early Stopping Logic
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("Generating predictions...")
    model.load_state_dict(best_model_wts)
    model.eval()

    preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            preds.extend(predicted.cpu().numpy())

    # Remap 0-6 back to 1-7
    final_preds = np.array(preds) + 1

    # 7. Save Submission
    os.makedirs("./submission", exist_ok=True)
    sub_df = pd.DataFrame({"Id": test_ids, "Cover_Type": final_preds})
    sub_df.to_csv("./submission/submission.csv", index=False)
    print(f"Submission saved to ./submission/submission.csv with {len(sub_df)} rows.")
