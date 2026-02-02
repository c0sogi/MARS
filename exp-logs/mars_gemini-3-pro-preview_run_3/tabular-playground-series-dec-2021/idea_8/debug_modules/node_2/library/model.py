import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
import library.config as config
import library.data_loader as data_loader

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------


class CrossNetV2(nn.Module):
    """
    Vector-based Deep Cross Network (DCN v2).
    Formula: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
    Where W_l is a weight vector (implemented via Linear(d, 1) broadcast).
    """

    def __init__(self, input_dim, num_layers):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            # Linear layer to compute the scalar/vector weight part.
            # In vector DCN, we typically learn a weight vector w.
            # x_l^T * w -> scalar (or broadcasted vector).
            # Efficient implementation: Linear(input_dim, input_dim) with diagonal constraint
            # OR Linear(input_dim, 1) if we assume scalar interaction per layer (DCN-Mix simplified)
            # OR standard DCN-V: W is diagonal matrix.
            # Here we implement the standard efficient DCN form: x_0 * (Linear(x_l)) + x_l
            # where Linear maps d -> d.
            self.layers.append(nn.Linear(input_dim, input_dim))

    def forward(self, x):
        x0 = x
        xi = x
        for layer in self.layers:
            # x_{l+1} = x0 * (W x_l + b) + x_l
            # Note: We use element-wise multiplication with x0
            interaction = x0 * layer(xi)
            xi = interaction + xi
        return xi


class ResNetBlock(nn.Module):
    """
    Standard Residual Block for Tabular Data.
    Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> Add -> ReLU
    """

    def __init__(self, hidden_dim, dropout_rate):
        super(ResNetBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class ParallelDCNResNet(nn.Module):
    """
    Hybrid architecture with parallel Deep ResNet and CrossNet branches.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_resnet_blocks,
        num_dcn_layers,
        dropout_rate,
        num_classes,
    ):
        super(ParallelDCNResNet, self).__init__()

        # Deep Branch
        self.deep_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )

        deep_blocks = []
        for _ in range(num_resnet_blocks):
            deep_blocks.append(ResNetBlock(hidden_dim, dropout_rate))
        self.deep_blocks = nn.Sequential(*deep_blocks)

        # Cross Branch
        self.cross_net = CrossNetV2(input_dim, num_dcn_layers)

        # Classification Head
        # Concatenate Deep (hidden_dim) and Cross (input_dim) outputs
        concat_dim = hidden_dim + input_dim
        self.final_linear = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # Deep Branch
        deep_out = self.deep_proj(x)
        deep_out = self.deep_blocks(deep_out)

        # Cross Branch
        cross_out = self.cross_net(x)

        # Concatenate
        combined = torch.cat([deep_out, cross_out], dim=1)

        # Output
        logits = self.final_linear(combined)
        return logits


# -----------------------------------------------------------------------------
# Training Pipeline
# -----------------------------------------------------------------------------


def train_model(train_loader, val_loader, device, epochs=config.EPOCHS):
    """
    Trains the ParallelDCNResNet model using SWA.
    """
    # 1. Determine Input Dimension from data
    # Get a sample batch to check dimensions
    sample_X, _ = next(iter(train_loader))
    input_dim = sample_X.shape[1]

    print(
        f"Initializing model with Input Dim: {input_dim}, Hidden Dim: {config.HIDDEN_DIM}"
    )

    model = ParallelDCNResNet(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        num_resnet_blocks=config.NUM_RESNET_BLOCKS,
        num_dcn_layers=config.NUM_DCN_LAYERS,
        dropout_rate=config.DROPOUT_RATE,
        num_classes=config.NUM_CLASSES,
    ).to(device)

    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler for Phase 1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # SWA Setup
    swa_model = AveragedModel(model)
    swa_start = config.SWA_START_EPOCH
    swa_scheduler = SWALR(optimizer, swa_lr=config.SWA_LR)

    best_val_acc = 0.0

    print("Starting training...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_loss = running_loss / total
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

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # SWA Logic
        if epoch >= swa_start:
            print(f"  -> SWA Update (Epoch {epoch})")
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step(val_acc)
            # Track best model for fallback (optional, but good practice)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # We could save the best non-SWA model here if needed

    # End of training: Update BN statistics for SWA model
    print("Updating SWA Batch Normalization statistics...")
    update_bn(train_loader, swa_model, device=device)

    # Save SWA model
    print(f"Saving SWA model to {config.MODEL_PATH}...")
    torch.save(swa_model.state_dict(), config.MODEL_PATH)

    return swa_model


# -----------------------------------------------------------------------------
# Submission Logic
# -----------------------------------------------------------------------------


def generate_submission(model, test_loader, device):
    """
    Generates predictions and saves to submission.csv.
    """
    print("Generating predictions on Test set...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            # Map 0-6 back to 1-7
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())

    # Load test IDs from metadata
    df_test = pd.read_parquet(config.TEST_PATH)
    ids = df_test[config.ID_COL].values

    # Create submission DataFrame
    submission = pd.DataFrame({config.ID_COL: ids, config.TARGET_COL: predictions})

    print(f"Saving submission to {config.SUBMISSION_PATH}...")
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------


def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    # Load Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        load_cached_data=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # Train
    model = train_model(train_loader, val_loader, device)

    # Generate Submission
    generate_submission(model, test_loader, device)
