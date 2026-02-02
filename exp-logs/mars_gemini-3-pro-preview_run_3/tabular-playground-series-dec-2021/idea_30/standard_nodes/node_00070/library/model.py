import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import pandas as pd
import numpy as np
import os

from library.config import (
    HIDDEN_DIM,
    DROPOUT,
    NUM_CLASSES,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SCHEDULER_MODE,
    PATIENCE,
    SUBMISSION_PATH,
    DEVICE,
)


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Warm-Start Initialization.
    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l
    """

    def __init__(self, in_features):
        super(VectorCrossLayer, self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.bias = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Warm-Start Initialization: Near-Zero Standard Deviation
        # This ensures the layer starts as an approximate identity mapping.
        nn.init.normal_(self.weight, mean=0, std=1e-4)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        """
        Args:
            x0: Initial input to the DCN stack (Batch, Features)
            xl: Input from the previous layer (Batch, Features)
        """
        # Compute dot product (x_l . w) per sample -> (Batch, 1)
        # We perform element-wise mult then sum over features
        score = torch.sum(xl * self.weight, dim=1, keepdim=True)

        # Apply formula: x0 * score + b + xl
        out = x0 * score + self.bias + xl
        return out


class PreActResNetBlock(nn.Module):
    """
    Full Pre-Activation ResNet Block.
    Topology: BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, dim, dropout=0.2):
        super(PreActResNetBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Block 1
        out = self.bn1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.lin1(out)

        # Block 2
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.lin2(out)

        # Residual connection
        return out + x


class ParallelDCNResNet(nn.Module):
    """
    Deep Parallel Vector-DCN-ResNet (Full Pre-Activation & Warm-Start).
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        num_resnet_blocks=4,
        num_cross_layers=4,
        dropout=0.2,
    ):
        super(ParallelDCNResNet, self).__init__()

        # Branch 1: Vector-based Deep & Cross Network (DCN)
        # Operates on the raw input dimension
        self.cross_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(num_cross_layers)]
        )

        # Branch 2: Deep Full Pre-Activation ResNet Backbone
        # Projects input to hidden_dim first
        self.resnet_proj = nn.Linear(input_dim, hidden_dim)
        self.resnet_blocks = nn.ModuleList(
            [PreActResNetBlock(hidden_dim, dropout) for _ in range(num_resnet_blocks)]
        )

        # Combination Head
        # Concatenates outputs of DCN (input_dim) and ResNet (hidden_dim)
        final_input_dim = input_dim + hidden_dim
        self.head = nn.Linear(final_input_dim, num_classes)

    def forward(self, x):
        # Branch 1: DCN
        # x0 is the fixed original input x
        x_dcn = x
        for layer in self.cross_layers:
            x_dcn = layer(x, x_dcn)

        # Branch 2: ResNet
        x_res = self.resnet_proj(x)
        for block in self.resnet_blocks:
            x_res = block(x_res)

        # Concatenate
        combined = torch.cat([x_dcn, x_res], dim=1)

        # Final Classification
        logits = self.head(combined)
        return logits


def train_model(model, train_loader, val_loader, device):
    """
    Trains the model using AdamW, ReduceLROnPlateau, and Early Stopping.
    """
    model.to(device)

    # Optimizer: Decoupled Weight Decay (AdamW)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Scheduler: Aggressive Decay on Plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=SCHEDULER_MODE,
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        avg_train_loss = train_loss / total
        train_acc = correct / total

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Reporting
        print(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.6f}, Train Acc: {train_acc:.6f}, Val Loss: {avg_val_loss:.6f}, Val Acc: {val_acc:.6f}"
        )

        # Scheduler Step
        scheduler.step(val_acc)

        # Early Stopping Check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Restore best model
    print(f"Restoring best model with Val Acc: {best_val_acc:.6f}")
    model.load_state_dict(best_model_state)
    return model


def predict_and_submit(model, test_loader, test_ids, device):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)

            # Map 0-indexed classes back to 1-indexed (1-7)
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())

    # Create submission DataFrame
    df_sub = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Save
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
