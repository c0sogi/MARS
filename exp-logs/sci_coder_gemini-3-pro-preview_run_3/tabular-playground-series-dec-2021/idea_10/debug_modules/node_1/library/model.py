import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import stochastic_depth
import numpy as np
import pandas as pd
import os
import copy
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders

# ------------------------------------------------------------------------------
# 1. Model Components
# ------------------------------------------------------------------------------


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, Channels)
        # Squeeze operation is identity for 1D vectors (Global Average Pooling not needed)
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(in_features)
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.BatchNorm1d(hidden_features)
        self.linear2 = nn.Linear(hidden_features, in_features)

        self.se = SEBlock(in_features)
        self.drop_path_prob = drop_path

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.norm2(out)
        out = self.linear2(out)

        out = self.se(out)

        if self.training and self.drop_path_prob > 0.0:
            out = stochastic_depth(
                out, self.drop_path_prob, mode="batch", training=True
            )

        return identity + out


class DCNv2Vector(nn.Module):
    """
    Vector-based Deep & Cross Network Layer.
    Implements Dot-Product Mixing: x_{l+1} = x_0 * (x_l^T w) + b + x_l
    """

    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim

        self.W = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim)) for _ in range(num_layers)]
        )
        self.b = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)]
        )

        for w in self.W:
            nn.init.xavier_normal_(w.unsqueeze(0))

    def forward(self, x):
        x0 = x
        xl = x
        for i in range(self.num_layers):
            # dot: (Batch, D) * (D,) -> (Batch, D) -> sum -> (Batch, 1)
            dot = (xl * self.W[i]).sum(dim=1, keepdim=True)
            xl = x0 * dot + self.b[i] + xl
        return xl


class ParallelDCN_SE_ResNet(nn.Module):
    def __init__(self, input_dim, num_classes=7):
        super().__init__()

        # Branch 1: DCN (Explicit Interactions)
        self.dcn = DCNv2Vector(input_dim, num_layers=3)

        # Branch 2: SE-ResNet (Deep Implicit Features)
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU()
        )

        # Stacking Residual Blocks with Stochastic Depth
        self.blocks = nn.ModuleList(
            [ResBlock(512, 512, drop_path=0.1) for _ in range(3)]
        )

        # Combination Head
        self.head = nn.Sequential(
            nn.BatchNorm1d(input_dim + 512), nn.Linear(input_dim + 512, num_classes)
        )

    def forward(self, x):
        # Branch 1
        x_dcn = self.dcn(x)

        # Branch 2
        x_res = self.stem(x)
        for block in self.blocks:
            x_res = block(x_res)

        # Combine
        x_cat = torch.cat([x_dcn, x_res], dim=1)
        out = self.head(x_cat)
        return out


# ------------------------------------------------------------------------------
# 2. Training & Orchestration
# ------------------------------------------------------------------------------


def train_and_predict(epochs=60, batch_size=4096, quick_run=False):
    """
    Orchestrates the training of ParallelDCN_SE_ResNet and generates submission.
    """
    seed_everything(42)
    device = get_device()
    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size, quick_run=quick_run, cache_dir="./working/idea_10/"
    )

    # Determine input dimension from a batch
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]
    num_classes = 7  # 7 Cover Types

    # Initialize Model
    model = ParallelDCN_SE_ResNet(input_dim, num_classes).to(device)

    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 8
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
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

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Print full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} Acc: {train_acc} | Val Loss: {val_loss} Acc: {val_acc}"
        )

        scheduler.step(val_acc)

        # Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Inference
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

    # Save Submission
    os.makedirs("./submission", exist_ok=True)
    sub_df = pd.DataFrame({"Id": test_ids, "Cover_Type": final_preds})
    sub_df.to_csv("./submission/submission.csv", index=False)
    print(f"Submission saved to ./submission/submission.csv with {len(sub_df)} rows.")
