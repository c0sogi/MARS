import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import os
import pandas as pd

from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders


class LowRankCrossLayer(nn.Module):
    """
    Low-Rank Factorized Cross Layer.
    Decomposes the interaction matrix W into U and V such that W = U @ V.T.
    Formula: x_{l+1} = x_0 * ( (x_l @ V) @ U.T + b ) + x_l
    """

    def __init__(self, input_dim, rank=16):
        super(LowRankCrossLayer, self).__init__()
        self.input_dim = input_dim
        self.rank = rank

        # V: Projects input_dim -> rank
        self.V = nn.Parameter(torch.Tensor(input_dim, rank))
        # U: Projects rank -> input_dim
        self.U = nn.Parameter(torch.Tensor(input_dim, rank))
        # Bias: input_dim
        self.bias = nn.Parameter(torch.Tensor(input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize U and V with Xavier Uniform to maintain variance
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.U)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        """
        Args:
            x0: Initial input features (Batch, Input_Dim)
            xl: Output from previous layer (Batch, Input_Dim)
        """
        # Compute (xl @ V) @ U.T
        # xl: (B, D), V: (D, R) -> proj: (B, R)
        proj = torch.matmul(xl, self.V)
        # proj: (B, R), U.T: (R, D) -> reconstruction: (B, D)
        reconstruction = torch.matmul(proj, self.U.t())

        # Add bias
        interaction = reconstruction + self.bias

        # Element-wise multiply with x0 and add residual xl
        out = x0 * interaction + xl
        return out


class ResNetBlock(nn.Module):
    """
    Standard Wide ResNet Block for tabular data.
    Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout
    """

    def __init__(self, hidden_dim, dropout_rate):
        super(ResNetBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        return out + residual


class ParallelDCNResNet(nn.Module):
    """
    Parallel Low-Rank DCN-ResNet Architecture.
    Branch 1: Stack of LowRankCrossLayers (DCN).
    Branch 2: Wide ResNet Backbone.
    Head: Concatenation -> Linear.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        dcn_rank=16,
        resnet_hidden=512,
        resnet_blocks=2,
        dropout=0.2,
        dcn_layers=3,
    ):
        super(ParallelDCNResNet, self).__init__()

        # --- DCN Branch ---
        # Stack of Low-Rank Cross Layers
        self.dcn_layers = nn.ModuleList(
            [LowRankCrossLayer(input_dim, dcn_rank) for _ in range(dcn_layers)]
        )

        # --- ResNet Branch ---
        # Projection to hidden dimension
        self.resnet_projection = nn.Sequential(
            nn.Linear(input_dim, resnet_hidden),
            nn.BatchNorm1d(resnet_hidden),
            nn.ReLU(),
        )
        # Residual Blocks
        self.resnet_blocks = nn.Sequential(
            *[ResNetBlock(resnet_hidden, dropout) for _ in range(resnet_blocks)]
        )

        # --- Combination Head ---
        # Concatenate DCN output (input_dim) and ResNet output (resnet_hidden)
        self.head = nn.Linear(input_dim + resnet_hidden, num_classes)

    def forward(self, x):
        # DCN Branch
        x_dcn = x
        for layer in self.dcn_layers:
            x_dcn = layer(x, x_dcn)  # x is x0, x_dcn is xl

        # ResNet Branch
        x_res = self.resnet_projection(x)
        x_res = self.resnet_blocks(x_res)

        # Combine
        combined = torch.cat([x_dcn, x_res], dim=1)
        logits = self.head(combined)

        return logits


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def run_training():
    """
    Executes the training pipeline, including data loading, model initialization,
    training loop with early stopping, and saving the best model.
    """
    seed_everything(Config.SEED)
    device = get_device()

    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # Determine input dimension from data
    # train_loader.dataset is a ForestCoverDataset, which has .X attribute
    input_dim = train_loader.dataset.X.shape[1]
    num_classes = Config.NUM_CLASSES

    print(f"Data Loaded. Input Dimension: {input_dim}, Num Classes: {num_classes}")

    # Initialize Model
    model = ParallelDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        dcn_rank=Config.DCN_RANK,
        resnet_hidden=Config.RESNET_HIDDEN_DIM,
        resnet_blocks=Config.RESNET_NUM_BLOCKS,
        dropout=Config.DROPOUT_RATE,
    ).to(device)

    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=Config.SCHEDULER_MODE,
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # Training Loop with Early Stopping
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, "
            f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}"
        )

        # Scheduler step
        scheduler.step(val_acc)

        # Early Stopping Check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save best model immediately to disk as well
            torch.save(best_model_wts, Config.MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training Complete. Best Validation Accuracy: {best_val_acc:.6f}")

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model, test_loader, test_ids


def generate_submission(model, test_loader, test_ids):
    """
    Generates predictions using the trained model and saves to submission.csv.
    """
    device = get_device()
    model.eval()
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            predictions.extend(preds.cpu().numpy())

    # Map predictions back to original labels using Inverse Map
    final_preds = [Config.INVERSE_LABEL_MAP[p] for p in predictions]

    # Create DataFrame
    df_sub = pd.DataFrame({"Id": test_ids, "Cover_Type": final_preds})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved.")


def train_and_predict():
    """
    Orchestrates the full pipeline: Training -> Inference -> Submission.
    """
    model, test_loader, test_ids = run_training()
    generate_submission(model, test_loader, test_ids)
