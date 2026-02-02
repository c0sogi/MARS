import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from library.config import Config

# ==========================================
# Model Architecture
# ==========================================


class LowRankCrossLayer(nn.Module):
    """
    Asymmetric Low-Rank Factorized Cross Layer.
    Formula: x_{l+1} = x_0 * (U * V^T * x_l + b) + x_l
    """

    def __init__(self, input_dim, rank=4):
        super(LowRankCrossLayer, self).__init__()
        self.input_dim = input_dim
        self.rank = rank

        # Factorized Weight Matrix W = U * V^T
        # V projects input -> rank space
        self.V = nn.Parameter(torch.Tensor(rank, input_dim))
        # U projects rank -> input space
        self.U = nn.Parameter(torch.Tensor(input_dim, rank))

        self.bias = nn.Parameter(torch.zeros(input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Warm-Start Initialization: Near-Zero Std Dev
        # Ensures the layer starts as an approximate identity mapping
        nn.init.normal_(self.U, mean=0, std=1e-4)
        nn.init.normal_(self.V, mean=0, std=1e-4)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        # x0, xl: [batch_size, input_dim]

        # 1. Project down to rank space: (batch, rank) = xl @ V.T
        projected = F.linear(xl, self.V)

        # 2. Project back to input space: (batch, input_dim) = projected @ U.T
        reprojected = F.linear(projected, self.U)

        # 3. Apply DCN formula
        return x0 * (reprojected + self.bias) + xl


class PreActResBlock(nn.Module):
    """
    Full Pre-Activation Residual Block.
    Structure: BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, dim, dropout=0.2):
        super(PreActResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(dim, dim)

        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.linear1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.linear2(out)

        return out + residual


class DeepSupervisedHybridModel(nn.Module):
    """
    Deeply-Supervised Asymmetric Parallel Low-Rank-DCN-ResNet.
    """

    def __init__(self, input_dim, num_classes=7, hidden_dim=512):
        super(DeepSupervisedHybridModel, self).__init__()

        # Hyperparameters from Config
        rank = Config.DCN_RANK
        dcn_layers = Config.DCN_LAYERS
        resnet_blocks = Config.RESNET_BLOCKS
        dropout = Config.DROPOUT

        # Branch 1: Low-Rank DCN (Asymmetric Depth)
        self.dcn_layers = nn.ModuleList(
            [LowRankCrossLayer(input_dim, rank=rank) for _ in range(dcn_layers)]
        )

        # Branch 2: Deep ResNet Backbone
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [PreActResBlock(hidden_dim, dropout=dropout) for _ in range(resnet_blocks)]
        )

        # Auxiliary Head (Attached after Block 3 / Index 2)
        self.aux_head = nn.Linear(hidden_dim, num_classes)

        # Combination Head
        # Concatenates Input-Dim DCN output with Hidden-Dim ResNet output
        self.final_linear = nn.Linear(input_dim + hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch, input_dim]

        # --- Branch 1: DCN ---
        x_dcn = x
        for layer in self.dcn_layers:
            x_dcn = layer(x, x_dcn)

        # --- Branch 2: ResNet ---
        x_res = self.input_proj(x)
        aux_logits = None

        for i, block in enumerate(self.blocks):
            x_res = block(x_res)

            # Capture Auxiliary Output after Block 3 (index 2)
            if i == 2:
                aux_logits = self.aux_head(x_res)

        # --- Combination ---
        combined = torch.cat([x_dcn, x_res], dim=1)
        primary_logits = self.final_linear(combined)

        return primary_logits, aux_logits


# ==========================================
# Training & Inference Logic
# ==========================================


def train_model(train_loader, val_loader):
    """
    Trains the DeepSupervisedHybridModel with Annealed Multi-Loss Optimization.
    """
    device = Config.DEVICE

    # Performance Tuning
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Determine Input Dimension from Data
    dummy_x, _ = next(iter(train_loader))
    input_dim = dummy_x.shape[1]

    print(f"Initializing model with Input Dim: {input_dim}, Hidden Dim: 512")
    model = DeepSupervisedHybridModel(
        input_dim, num_classes=Config.NUM_CLASSES, hidden_dim=512
    ).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )
    criterion = nn.CrossEntropyLoss()

    # Early Stopping State
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stop_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs on {device}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Calculate Annealed Lambda for Auxiliary Loss
        # Linearly decay from START to END
        progress = epoch / Config.EPOCHS
        lambda_t = (
            Config.AUX_LOSS_WEIGHT_START
            - (Config.AUX_LOSS_WEIGHT_START - Config.AUX_LOSS_WEIGHT_END) * progress
        )
        lambda_t = max(0.0, lambda_t)

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward Pass
            prim_logits, aux_logits = model(inputs)

            # Multi-Loss Calculation
            loss_prim = criterion(prim_logits, labels)
            loss_aux = criterion(aux_logits, labels)

            loss = loss_prim + lambda_t * loss_aux

            # Backward Pass
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(prim_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward (Aux ignored in inference/val)
                prim_logits, _ = model(inputs)
                loss = criterion(prim_logits, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(prim_logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {epoch_loss} Acc: {epoch_acc} | Val Loss: {val_loss} Val Acc: {val_acc}"
        )

        # Scheduler Step
        scheduler.step(val_acc)

        # Early Stopping Check
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # Restore Best Model
    print(f"Training complete. Best Validation Accuracy: {best_acc}")
    model.load_state_dict(best_model_wts)

    # Save checkpoint to working dir
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Best model saved to {save_path}")

    return model


def generate_submission(model, test_loader, test_ids):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    device = Config.DEVICE
    model.eval()
    predictions = []

    print("Generating predictions on test set...")

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)

            prim_logits, _ = model(inputs)
            _, preds = torch.max(prim_logits, 1)

            # Map 0-6 back to original 1-7 class labels
            preds = preds + 1
            predictions.extend(preds.cpu().numpy())

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    # Save
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
