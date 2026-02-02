import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.data_utils import get_dataloaders

# --------------------------------------------------------------------------
# Model Architecture
# --------------------------------------------------------------------------


class VectorCrossLayer(nn.Module):
    """
    Implements the Vector-based Cross Layer (Rank-1).
    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l
    """

    def __init__(self, input_dim):
        super(VectorCrossLayer, self).__init__()
        # Parameter w: weight vector for the dot product
        self.w = nn.Parameter(torch.empty(input_dim))
        # Parameter b: bias vector
        self.b = nn.Parameter(torch.empty(input_dim))
        self._init_parameters()

    def _init_parameters(self):
        # Initialize w using Xavier Uniform and b to zeros
        nn.init.xavier_uniform_(self.w.unsqueeze(0))
        nn.init.zeros_(self.b)

    def forward(self, x0, xl):
        """
        Args:
            x0: The original input features (batch_size, input_dim)
            xl: The output of the previous layer (batch_size, input_dim)
        """
        # Compute scalar projection (x_l . w) -> (batch_size, 1)
        projection = torch.matmul(xl, self.w).unsqueeze(1)

        # Interaction: x0 * scalar_projection (broadcasted)
        interaction = x0 * projection

        # Output: interaction + bias + residual
        return interaction + self.b + xl


class VectorDCN(nn.Module):
    """
    Stack of VectorCrossLayers.
    """

    def __init__(self, input_dim, num_layers=2):
        super(VectorDCN, self).__init__()
        self.layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(num_layers)]
        )

    def forward(self, x):
        x0 = x
        xl = x
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl


class ResNetBlock(nn.Module):
    """
    Wide ResNet Block with Residual Connection.
    Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> Dropout -> Add -> ReLU
    """

    def __init__(self, hidden_dim, dropout_rate):
        super(ResNetBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)

        # Residual connection
        out = out + residual
        out = self.activation(out)
        out = self.dropout(out)
        return out


class ParallelVectorDCNResNet(nn.Module):
    """
    Hybrid architecture: Parallel Vector-DCN and Wide ResNet.
    """

    def __init__(
        self,
        input_dim,
        num_classes=Config.NUM_CLASSES,
        hidden_dim=Config.HIDDEN_DIM,
        resnet_blocks=Config.RESNET_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
        num_cross_layers=2,
    ):
        super(ParallelVectorDCNResNet, self).__init__()

        # Branch 1: Vector DCN (keeps dimension as input_dim)
        self.dcn = VectorDCN(input_dim, num_layers=num_cross_layers)

        # Branch 2: Wide ResNet (projects to hidden_dim)
        self.resnet_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )
        self.resnet_blocks = nn.ModuleList(
            [ResNetBlock(hidden_dim, dropout_rate) for _ in range(resnet_blocks)]
        )

        # Combination Head
        # Concatenates DCN output (input_dim) and ResNet output (hidden_dim)
        concat_dim = input_dim + hidden_dim

        self.head = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),
            nn.BatchNorm1d(concat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(concat_dim // 2, num_classes),
        )

    def forward(self, x):
        # Branch 1: DCN
        x_dcn = self.dcn(x)

        # Branch 2: ResNet
        x_res = self.resnet_proj(x)
        for block in self.resnet_blocks:
            x_res = block(x_res)

        # Concatenate
        x_cat = torch.cat([x_dcn, x_res], dim=1)

        # Final Classification
        logits = self.head(x_cat)
        return logits


# --------------------------------------------------------------------------
# Training & Evaluation Logic
# --------------------------------------------------------------------------


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


def train_and_evaluate(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=0
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save best model to disk
            torch.save(
                best_model_wts, os.path.join(Config.WORKING_DIR, "best_model.pth")
            )
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Accuracy: {best_acc:.6f}")

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model


def generate_submission(model, test_loader, test_ids, device):
    model.eval()
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            # Convert 0-6 class indices back to 1-7 target labels
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())

    # Create DataFrame
    submission = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: predictions})

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_experiment():
    # Setup
    Config.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader, input_dim, test_ids = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Model
    model = ParallelVectorDCNResNet(
        input_dim=input_dim,
        num_classes=Config.NUM_CLASSES,
        hidden_dim=Config.HIDDEN_DIM,
        resnet_blocks=Config.RESNET_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
        num_cross_layers=2,  # Defaulting to 2 layers for the DCN branch
    ).to(device)

    # Train
    model = train_and_evaluate(model, train_loader, val_loader, device)

    # Predict
    generate_submission(model, test_loader, test_ids, device)
