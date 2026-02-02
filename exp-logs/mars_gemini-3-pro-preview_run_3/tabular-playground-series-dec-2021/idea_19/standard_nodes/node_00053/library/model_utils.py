import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import copy
import time
from library.config import Config

# ==========================================
# 1. Model Architecture
# ==========================================


class LowRankCrossLayer(nn.Module):
    """
    Implements the Low-Rank Factorized Cross Layer:
    x_{l+1} = x_0 * (U (V^T x_l) + b) + x_l

    Where:
    - U, V are matrices of shape (input_dim, rank)
    - * denotes element-wise multiplication
    """

    def __init__(self, input_dim, rank=16):
        super(LowRankCrossLayer, self).__init__()
        self.input_dim = input_dim
        self.rank = rank

        # Initialize parameters
        # U and V decompose the weight matrix W (d x d) into (d x r) and (r x d)
        self.U = nn.Parameter(torch.Tensor(input_dim, rank))
        self.V = nn.Parameter(torch.Tensor(input_dim, rank))
        self.bias = nn.Parameter(torch.Tensor(input_dim))

        # Init weights
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        # x0: (batch, input_dim) - original input features
        # xl: (batch, input_dim) - output of previous layer

        # Compute V^T * xl -> (batch, rank)
        # Note: Linear(x) computes xA^T. Here we want xV.
        # x is (B, D), V is (D, R). x @ V -> (B, R)
        vt_xl = torch.matmul(xl, self.V)

        # Compute U * (V^T xl) -> (batch, input_dim)
        # vt_xl is (B, R), U is (D, R). We want (B, R) @ U^T -> (B, D)
        # U^T is (R, D).
        interaction = torch.matmul(vt_xl, self.U.t())

        # Add bias
        interaction = interaction + self.bias

        # Element-wise multiplication with x0 and residual connection
        out = x0 * interaction + xl
        return out


class ResidualBlock(nn.Module):
    """
    Standard Residual Block for the Wide ResNet backbone.
    Linear -> BN -> ReLU -> Linear -> BN -> Add -> ReLU
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)
        return out


class ParallelDCNResNet(nn.Module):
    """
    Hybrid Architecture:
    1. Low-Rank DCN Branch: Explicit feature interactions.
    2. Wide ResNet Branch: Deep implicit representation learning.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=512,
        low_rank_factor=16,
        num_cross_layers=3,
        num_res_blocks=3,
        dropout=0.1,
    ):
        super(ParallelDCNResNet, self).__init__()

        # --- Branch 1: Low-Rank DCN ---
        self.num_cross_layers = num_cross_layers
        self.cross_layers = nn.ModuleList(
            [
                LowRankCrossLayer(input_dim, low_rank_factor)
                for _ in range(num_cross_layers)
            ]
        )

        # --- Branch 2: Wide ResNet ---
        self.res_project = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
        )

        # --- Combination Head ---
        # Concatenate DCN output (input_dim) + ResNet output (hidden_dim)
        concat_dim = input_dim + hidden_dim
        self.final_bn = nn.BatchNorm1d(concat_dim)
        self.classifier = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # x: (batch, input_dim)

        # 1. DCN Branch
        x_dcn = x
        for layer in self.cross_layers:
            x_dcn = layer(x, x_dcn)

        # 2. ResNet Branch
        x_res = self.res_project(x)
        x_res = F.relu(x_res)
        for block in self.res_blocks:
            x_res = block(x_res)

        # 3. Combination
        combined = torch.cat([x_dcn, x_res], dim=1)
        combined = self.final_bn(combined)
        logits = self.classifier(combined)

        return logits


# ==========================================
# 2. Training Utility
# ==========================================


def train_model(train_loader, val_loader, input_dim, num_classes):
    """
    Trains the ParallelDCNResNet model using Budget-Aware Cosine Optimization.
    """
    device = torch.device(Config.DEVICE)
    print(f"Training on device: {device}")

    # Initialize Model
    model = ParallelDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=Config.HIDDEN_DIM,
        low_rank_factor=Config.LOW_RANK_FACTOR,
        num_cross_layers=3,  # Fixed architecture depth
        num_res_blocks=3,
        dropout=Config.DROPOUT,
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Scheduler: Cosine Annealing over fixed epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_state = None

    start_time = time.time()

    for epoch in range(Config.EPOCHS):
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

        train_loss /= total
        train_acc = correct / total

        # Validation
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

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Checkpointing (Deepcopy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

    total_time = time.time() - start_time
    print(
        f"Training complete in {total_time:.2f}s. Best Validation Accuracy: {best_val_acc:.6f}"
    )

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


# ==========================================
# 3. Inference Utility
# ==========================================


def predict_and_submit(model, test_loader, test_ids, submission_path):
    """
    Generates predictions and saves submission file.
    """
    device = torch.device(Config.DEVICE)
    model.eval()

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            # Map 0-6 back to 1-7
            predicted_classes = predicted.cpu().numpy() + 1
            predictions.extend(predicted_classes)

    predictions = np.array(predictions)

    # Verify lengths
    if len(predictions) != len(test_ids):
        print(
            f"Warning: Number of predictions ({len(predictions)}) does not match number of IDs ({len(test_ids)})."
        )

    # Create DataFrame
    df_sub = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    # Save
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
