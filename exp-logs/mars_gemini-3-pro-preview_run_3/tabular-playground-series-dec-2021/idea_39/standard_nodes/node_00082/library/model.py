import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import copy
from library.utils import get_device, seed_everything
from library.data_loader import get_dataloaders

# ==========================================
# Component Layers
# ==========================================


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l
    where (.) denotes dot product resulting in a scalar (per sample),
    and (*) denotes element-wise multiplication.
    """

    def __init__(self, input_dim):
        super(VectorCrossLayer, self).__init__()
        # Weight vector w: initialized with near-zero std to start as identity
        self.w = nn.Parameter(torch.empty(input_dim))
        # Bias vector b
        self.b = nn.Parameter(torch.empty(input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialization: N(0, 1e-4) for weights to ensure stability
        nn.init.normal_(self.w, mean=0, std=1e-4)
        nn.init.zeros_(self.b)

    def forward(self, x0, xl):
        """
        Args:
            x0: Initial input features (Batch, Dim)
            xl: Output from previous layer (Batch, Dim)
        """
        # Calculate scalar score per sample: x_l^T w
        # (Batch, Dim) * (Dim) -> (Batch, Dim) -> sum -> (Batch, 1)
        score = torch.sum(xl * self.w, dim=1, keepdim=True)

        # x_{l+1} = x_0 * score + b + x_l
        out = x0 * score + self.b + xl
        return out


class PreActResNetBlock(nn.Module):
    """
    Full Pre-Activation ResNet Block with Wide Topology.
    Structure: BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, dim, dropout=0.3):
        super(PreActResNetBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.lin1 = nn.Linear(dim, dim)

        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout2 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x):
        # Pre-activation: BN -> ReLU
        out = self.bn1(x)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.lin1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.lin2(out)

        # Residual connection
        return out + x


# ==========================================
# Main Model Architecture
# ==========================================


class WideAsymmetricDCNResNet(nn.Module):
    """
    Stabilized Wide Asymmetric Parallel Vector-DCN-ResNet.

    Branch 1: Asymmetric Vector-Based DCN (3 Layers)
    Branch 2: Wide Full Pre-Activation ResNet (4 Blocks, Width 1024)
    """

    def __init__(self, input_dim, num_classes, hidden_dim=1024, dropout=0.3):
        super(WideAsymmetricDCNResNet, self).__init__()

        # --- Branch 1: DCN ---
        # 3 Cross Layers
        self.cross_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(3)]
        )

        # --- Branch 2: Wide ResNet ---
        # Projection to hidden width (1024)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 4 Pre-Activation Blocks
        self.res_blocks = nn.ModuleList(
            [PreActResNetBlock(hidden_dim, dropout=dropout) for _ in range(4)]
        )

        # --- Combination Head ---
        # Concatenates DCN output (input_dim) and ResNet output (hidden_dim)
        self.final_linear = nn.Linear(input_dim + hidden_dim, num_classes)

    def forward(self, x):
        # Branch 1: DCN
        x_dcn = x
        for layer in self.cross_layers:
            x_dcn = layer(x, x_dcn)  # Pass x0 and xl

        # Branch 2: ResNet
        x_res = self.input_proj(x)
        for block in self.res_blocks:
            x_res = block(x_res)

        # Concatenate
        concat = torch.cat([x_dcn, x_res], dim=1)

        # Classification
        logits = self.final_linear(concat)
        return logits


# ==========================================
# Training & Execution Logic
# ==========================================


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

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
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / total, correct / total


def predict_test(model, loader, device):
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for inputs, ids in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_ids.extend(ids.numpy())

    return all_ids, all_preds


def run_training(
    epochs=60,
    batch_size=4096,
    learning_rate=1e-3,
    warmup_epochs=5,
    patience=15,
    output_dir="./submission",
    cache_dir="./working/idea_39/",
):
    """
    Main execution function to train the model and generate submission.
    """
    seed_everything(42)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader, input_dim, num_classes = get_dataloaders(
        batch_size=batch_size, load_cached_data=True, cache_dir=cache_dir
    )
    print(f"Input Dim: {input_dim}, Num Classes: {num_classes}")

    # 2. Initialize Model
    model = WideAsymmetricDCNResNet(
        input_dim=input_dim, num_classes=num_classes, hidden_dim=1024, dropout=0.3
    ).to(device)

    # 3. Optimizer & Loss
    # AdamW with Decoupled Weight Decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    # Scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=5
    )

    # 4. Training Loop
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        # Linear Warmup Logic
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            current_lr = learning_rate * warmup_factor
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler Step (skip during warmup)
        if epoch >= warmup_epochs:
            scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} Acc: {val_acc:.6f}"
        )

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Accuracy: {best_acc:.6f}")

    # 5. Prediction
    print("Generating predictions...")
    model.load_state_dict(best_model_wts)
    ids, preds = predict_test(model, test_loader, device)

    # 6. Submission
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "submission.csv")

    # Map 0-indexed predictions back to 1-indexed labels if necessary
    # The dataset utils map 1-7 to 0-6. We need to map back 0-6 to 1-7.
    # Cover_Type is 1-7.
    preds_mapped = [p + 1 for p in preds]

    df_sub = pd.DataFrame({"Id": ids, "Cover_Type": preds_mapped})

    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
