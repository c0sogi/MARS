import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import copy
from library.config import Config
from library.data_utils import get_dataloaders

# ==========================================
# 1. Model Components
# ==========================================


class LowRankCrossLayer(nn.Module):
    """
    Low-Rank Cross Layer for DCN.
    Formula: x_{l+1} = x_0 * (U * (V^T * x_l)) + b + x_l
    Approximates the full weight matrix W with low-rank decomposition U*V^T.
    """

    def __init__(self, input_dim, rank):
        super(LowRankCrossLayer, self).__init__()
        self.input_dim = input_dim
        self.rank = rank

        # V: Projects input_dim -> rank (Represents V^T)
        self.linear_v = nn.Linear(input_dim, rank, bias=False)
        # U: Projects rank -> input_dim (Represents U)
        self.linear_u = nn.Linear(rank, input_dim, bias=False)

        # Bias vector
        self.bias = nn.Parameter(torch.zeros(input_dim))

        # Initialization
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.linear_u.weight)

    def forward(self, x_l, x_0):
        """
        Args:
            x_l: Output from previous layer (Batch, Input_Dim)
            x_0: Original input features (Batch, Input_Dim)
        """
        # Compute (V^T * x_l) -> Shape: (Batch, Rank)
        # Note: nn.Linear(in, out) computes x A^T. So this is x_l * V_weight^T.
        v_out = self.linear_v(x_l)

        # Compute U * (V^T * x_l) -> Shape: (Batch, Input_Dim)
        u_out = self.linear_u(v_out)

        # Interaction: x_0 * (U V^T x_l)
        interaction = x_0 * u_out

        # Add bias and residual connection
        out = interaction + self.bias + x_l
        return out


class ResNetBlock(nn.Module):
    """
    Wide ResNet Block for Tabular Data.
    Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> Dropout -> Add -> ReLU
    """

    def __init__(self, dim, dropout_rate):
        super(ResNetBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        out = out + residual
        out = F.relu(out)
        return out


class ParallelLowRankDCNResNet(nn.Module):
    """
    Hybrid architecture with parallel Low-Rank DCN and Wide ResNet branches.
    """

    def __init__(self):
        super(ParallelLowRankDCNResNet, self).__init__()

        input_dim = Config.INPUT_DIM
        num_classes = Config.NUM_CLASSES

        # Branch 1: Low-Rank DCN
        self.dcn_layers = nn.ModuleList(
            [
                LowRankCrossLayer(input_dim, Config.DCN_RANK)
                for _ in range(Config.DCN_LAYERS)
            ]
        )

        # Branch 2: Wide ResNet
        self.resnet_projection = nn.Linear(input_dim, Config.RESNET_WIDTH)
        self.resnet_bn = nn.BatchNorm1d(Config.RESNET_WIDTH)

        self.resnet_blocks = nn.ModuleList(
            [
                ResNetBlock(Config.RESNET_WIDTH, Config.RESNET_DROPOUT)
                for _ in range(Config.RESNET_DEPTH)
            ]
        )

        # Combination Head
        concat_dim = input_dim + Config.RESNET_WIDTH
        self.head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # x: (Batch, Input_Dim)

        # --- Branch 1: DCN ---
        x_dcn = x
        for layer in self.dcn_layers:
            x_dcn = layer(x_dcn, x)  # Pass current state and original input

        # --- Branch 2: ResNet ---
        # Project to width
        x_res = self.resnet_projection(x)
        x_res = self.resnet_bn(x_res)
        x_res = F.relu(x_res)

        # Apply blocks
        for block in self.resnet_blocks:
            x_res = block(x_res)

        # --- Combination ---
        x_combined = torch.cat([x_dcn, x_res], dim=1)
        logits = self.head(x_combined)

        return logits


# ==========================================
# 2. Training & Inference Logic
# ==========================================


def train_one_epoch(model, loader, optimizer, criterion, device):
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


def generate_submission(model, test_loader, device):
    print("Generating submission...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            # Map back from 0-6 to 1-7
            predicted_labels = predicted.cpu().numpy() + 1
            predictions.extend(predicted_labels)

    # Load Test IDs
    # We use the cached numpy file generated by data_utils
    test_ids_path = os.path.join(Config.WORKING_DIR, "test_ids.npy")
    if not os.path.exists(test_ids_path):
        raise FileNotFoundError(f"Test IDs not found at {test_ids_path}")

    test_ids = np.load(test_ids_path)

    # Create DataFrame
    df_sub = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: predictions})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


def run_pipeline():
    # 1. Setup
    print("Initializing pipeline...")
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    model = ParallelLowRankDCNResNet().to(device)

    # 4. Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_acc)

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss:.6f} Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} Acc: {val_acc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save checkpoint
            torch.save(best_model_wts, Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch+1}. Best Val Acc: {best_acc:.6f}"
            )
            break

    # 6. Final Inference
    print("Loading best model weights...")
    model.load_state_dict(best_model_wts)
    generate_submission(model, test_loader, device)


# Execute the pipeline
run_pipeline()
