import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import copy
import os
from library.config import Config


class LowRankCrossLayer(nn.Module):
    """
    Implements the Low-Rank Cross Layer for the DCN branch.
    Formula: x_{l+1} = x_0 * (U (V^T x_l) + b) + x_l
    Decomposes the interaction matrix W into U * V^T to reduce parameters.
    """

    def __init__(self, input_dim, rank):
        super(LowRankCrossLayer, self).__init__()
        self.input_dim = input_dim
        self.rank = rank

        # Low-rank matrices U and V
        # U: (d, r), V: (d, r)
        self.U = nn.Parameter(torch.Tensor(input_dim, rank))
        self.V = nn.Parameter(torch.Tensor(input_dim, rank))
        self.bias = nn.Parameter(torch.Tensor(input_dim))

        self.init_parameters()

    def init_parameters(self):
        # Xavier initialization for stability
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.zeros_(self.bias)

    def forward(self, x_l, x_0):
        """
        Args:
            x_l: Output from previous layer (Batch, input_dim)
            x_0: Original input features (Batch, input_dim)
        """
        # V^T x_l -> (Batch, rank)
        vt_xl = torch.matmul(x_l, self.V)

        # U (V^T x_l) -> (Batch, input_dim)
        u_vt_xl = torch.matmul(vt_xl, self.U.t())

        # Add bias
        interaction = u_vt_xl + self.bias

        # Element-wise multiply with x_0 and add residual
        out = x_0 * interaction + x_l
        return out


class ResNetBlock(nn.Module):
    """
    Wide ResNet Block with ReLU activations.
    Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> Add -> ReLU
    """

    def __init__(self, width, dropout):
        super(ResNetBlock, self).__init__()
        self.linear1 = nn.Linear(width, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)

        # Residual connection
        out += identity
        out = self.relu(out)

        return out


class ParallelLowRankDCNResNet(nn.Module):
    """
    Hybrid architecture combining Low-Rank DCN and Wide ResNet.
    """

    def __init__(self, input_dim, num_classes):
        super(ParallelLowRankDCNResNet, self).__init__()

        # --- Branch 1: Low-Rank DCN ---
        # We stack 3 layers to ensure sufficient interaction depth
        self.num_cross_layers = 3
        self.dcn_layers = nn.ModuleList(
            [
                LowRankCrossLayer(input_dim, Config.DCN_RANK)
                for _ in range(self.num_cross_layers)
            ]
        )

        # --- Branch 2: Wide ResNet Backbone ---
        # Project input to ResNet width
        self.resnet_proj = nn.Linear(input_dim, Config.RESNET_WIDTH)
        self.resnet_bn = nn.BatchNorm1d(Config.RESNET_WIDTH)
        self.resnet_relu = nn.ReLU()

        # Stack of Residual Blocks
        self.resnet_blocks = nn.Sequential(
            *[
                ResNetBlock(Config.RESNET_WIDTH, Config.RESNET_DROPOUT)
                for _ in range(Config.RESNET_LAYERS)
            ]
        )

        # --- Combination Head ---
        # Concatenate DCN output (input_dim) and ResNet output (RESNET_WIDTH)
        concat_dim = input_dim + Config.RESNET_WIDTH
        self.head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # x: (Batch, input_dim)

        # 1. DCN Branch
        x_dcn = x
        for layer in self.dcn_layers:
            x_dcn = layer(x_dcn, x)  # Pass current state and original input

        # 2. ResNet Branch
        x_res = self.resnet_proj(x)
        x_res = self.resnet_bn(x_res)
        x_res = self.resnet_relu(x_res)
        x_res = self.resnet_blocks(x_res)

        # 3. Combine
        x_concat = torch.cat([x_dcn, x_res], dim=1)
        logits = self.head(x_concat)

        return logits


def train_model(model, train_loader, val_loader):
    """
    Trains the model with Cosine Annealing and Early Stopping.
    """
    device = Config.DEVICE
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
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

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        # --- Validation Phase ---
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

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total

        # Step Scheduler
        scheduler.step()

        # Print metrics
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {epoch_loss:.6f} Acc: {epoch_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} Acc: {val_acc:.6f}"
        )

        # --- Early Stopping ---
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch+1}. Best Val Acc: {best_acc:.6f}"
            )
            break

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model


def predict_and_submit(model, test_loader, test_ids):
    """
    Generates predictions and saves submission CSV.
    """
    device = Config.DEVICE
    model.to(device)
    model.eval()

    predictions = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    # Convert predictions back to original class labels (0-6 -> 1-7)
    predictions = np.array(predictions) + 1

    # Create submission DataFrame
    df_sub = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: predictions})

    # Save to CSV
    save_path = Config.SUBMISSION_PATH
    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
