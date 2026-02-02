import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import os
from library.utils import seed_everything, get_device, save_submission
from library.data_processing import get_dataloaders


class VectorDCNLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Formula: x_{l+1} = x0 * (xl . w) + b + xl
    """

    def __init__(self, input_dim):
        super().__init__()
        # Weight vector w: initialized with Near-Zero Std (1e-4) to ensure warm-start
        self.weight = nn.Parameter(torch.randn(input_dim) * 1e-4)
        # Bias b: initialized to 0
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, xl):
        # xl . w is a dot product per sample -> (Batch, 1)
        # (Batch, Dim) * (Dim,) -> Sum over Dim -> (Batch, 1)
        mix = (xl * self.weight).sum(dim=1, keepdim=True)

        # Apply mixing to original input x0, add bias and residual xl
        return x0 * mix + self.bias + xl


class InvertedBottleneckBlock(nn.Module):
    """
    Inverted Bottleneck Block:
    Input -> BN -> Linear(512->2048) -> ReLU -> Dropout -> Linear(2048->512) -> Add(Input)
    """

    def __init__(self, dim=512, expansion_dim=2048, dropout=0.2):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, expansion_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(expansion_dim, dim)

        # Initialization: fc2 weights initialized with Near-Zero Std (1e-4)
        # This ensures the block starts as an identity mapping (resid + ~0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        resid = x
        out = self.bn(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out + resid


class AsymmetricParallelNet(nn.Module):
    """
    Asymmetric Parallel Vector-DCN-ResNet with Inverted Bottleneck Backbone.
    Branch 1: 3-Layer Vector DCN
    Branch 2: 4-Block Inverted Bottleneck ResNet (512 width)
    """

    def __init__(self, input_dim, num_classes=7):
        super().__init__()

        # Branch 1: Asymmetric Vector-Based DCN (3 Layers)
        self.dcn_layers = nn.ModuleList([VectorDCNLayer(input_dim) for _ in range(3)])

        # Branch 2: Deep Inverted Bottleneck ResNet Backbone
        # Project input to 512 dim for the backbone
        self.res_proj = nn.Linear(input_dim, 512)

        # 4 Blocks of Inverted Bottleneck
        self.resnet = nn.Sequential(
            *[
                InvertedBottleneckBlock(dim=512, expansion_dim=2048, dropout=0.2)
                for _ in range(4)
            ]
        )

        # Combination Head
        # Concatenate DCN output (input_dim) and ResNet output (512)
        self.head = nn.Linear(input_dim + 512, num_classes)

    def forward(self, x):
        # Branch 1: DCN
        # x is used as x0 (original input)
        x_dcn = x
        for layer in self.dcn_layers:
            x_dcn = layer(x, x_dcn)

        # Branch 2: ResNet
        x_res = self.res_proj(x)
        x_res = self.resnet(x_res)

        # Combine
        combined = torch.cat([x_dcn, x_res], dim=1)
        logits = self.head(combined)
        return logits


def train_model(
    model, train_loader, val_loader, device, epochs=60, lr=1e-3, patience=10
):
    """
    Training loop with AdamW, ReduceLROnPlateau, and Early Stopping.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=5, verbose=True
    )

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {epoch_loss:.6f}, Train Acc: {epoch_acc:.6f} - "
            f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}"
        )

        # Scheduler step
        scheduler.step(val_acc)

        # Early Stopping & Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Accuracy: {best_acc:.6f}")
    model.load_state_dict(best_model_wts)
    return model


def predict(model, test_loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # Shift back to 1-7 range (model predicts 0-6)
            predicted = predicted + 1
            preds.extend(predicted.cpu().numpy())

    return np.array(preds)


def main():
    """
    Main execution function.
    """
    # Setup
    seed_everything(42, deterministic=False)
    device = get_device()

    # Data
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=4096, load_cached_data=True, num_workers=4
    )

    # Model Init
    # Get input dim from a batch
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[1]
    num_classes = 7  # Classes 0-6 (mapped from 1-7)

    print(f"Initializing model with Input Dim: {input_dim}, Classes: {num_classes}")
    model = AsymmetricParallelNet(input_dim, num_classes).to(device)

    # Train
    model = train_model(
        model, train_loader, val_loader, device, epochs=60, lr=1e-3, patience=15
    )

    # Predict
    print("Generating predictions...")
    predictions = predict(model, test_loader, device)

    # Save
    save_submission(predictions, test_ids, "./submission/submission.csv")
