import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import copy
from library.config import Config


# --------------------------------------------------------------------------
# 1. Low-Rank Cross Layer
# --------------------------------------------------------------------------
class LowRankCrossLayer(nn.Module):
    """
    Implements the Low-Rank Factorized Cross Layer.
    Formula: x_{l+1} = x_0 * (U (V^T x_l) + b) + x_l
    """

    def __init__(self, in_features, rank):
        super(LowRankCrossLayer, self).__init__()
        self.in_features = in_features
        self.rank = rank

        # Factorized weight matrices U and V
        # U: (d, r), V: (d, r)
        self.U = nn.Parameter(torch.Tensor(in_features, rank))
        self.V = nn.Parameter(torch.Tensor(in_features, rank))
        self.bias = nn.Parameter(torch.Tensor(in_features))

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization for stability
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.zeros_(self.bias)

    def forward(self, x0, x):
        """
        Args:
            x0: Initial features (Batch, In_Features) - preserved for residual connection
            x: Current layer features (Batch, In_Features)
        """
        # Linear projection down: V^T x -> (Batch, Rank)
        # x @ V is (Batch, Rank)
        v_x = torch.matmul(x, self.V)

        # Linear projection up: U (V^T x) -> (Batch, In_Features)
        # v_x @ U.T is (Batch, In_Features)
        u_v_x = torch.matmul(v_x, self.U.t())

        # Element-wise interaction + Residual
        out = x0 * (u_v_x + self.bias) + x
        return out


# --------------------------------------------------------------------------
# 2. ResNet Block
# --------------------------------------------------------------------------
class ResNetBlock(nn.Module):
    """
    Standard Wide ResNet Block for Tabular Data.
    Linear -> BN -> ReLU -> Linear -> BN -> Add -> ReLU
    """

    def __init__(self, width):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(width, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.fc2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


# --------------------------------------------------------------------------
# 3. Parallel Low-Rank DCN-ResNet
# --------------------------------------------------------------------------
class ParallelDCNResNet(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        dcn_rank=Config.DCN_RANK,
        resnet_width=Config.RESNET_WIDTH,
        dcn_layers=3,
        resnet_blocks=2,
    ):
        super(ParallelDCNResNet, self).__init__()

        # Branch 1: Low-Rank DCN
        # Stack of LowRankCrossLayers
        self.dcn_layers = nn.ModuleList(
            [LowRankCrossLayer(input_dim, dcn_rank) for _ in range(dcn_layers)]
        )

        # Branch 2: Wide ResNet Backbone
        # Initial projection to hidden width
        self.resnet_projection = nn.Sequential(
            nn.Linear(input_dim, resnet_width), nn.BatchNorm1d(resnet_width), nn.ReLU()
        )
        # Residual Blocks
        self.resnet_blocks = nn.Sequential(
            *[ResNetBlock(resnet_width) for _ in range(resnet_blocks)]
        )

        # Combination Head
        # Concatenates DCN output (input_dim) and ResNet output (resnet_width)
        self.head = nn.Linear(input_dim + resnet_width, num_classes)

    def forward(self, x):
        # Branch 1: DCN
        x_dcn = x
        x0 = x
        for layer in self.dcn_layers:
            x_dcn = layer(x0, x_dcn)

        # Branch 2: ResNet
        x_res = self.resnet_projection(x)
        x_res = self.resnet_blocks(x_res)

        # Concatenate
        x_combined = torch.cat([x_dcn, x_res], dim=1)

        # Classification
        logits = self.head(x_combined)
        return logits


# --------------------------------------------------------------------------
# 4. Training Function
# --------------------------------------------------------------------------
def train_model(model, train_loader, val_loader):
    """
    Trains the model using Cosine Annealing and Early Stopping.
    """
    device = Config.DEVICE
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=0
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = Config.PATIENCE
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

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

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {epoch_loss:.6f} Acc: {epoch_acc:.6f} | "
            f"Val Loss: {val_epoch_loss:.6f} Acc: {val_epoch_acc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save checkpoint
            torch.save(best_model_wts, Config.MODEL_CHECKPOINT_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val Acc: {best_val_acc:.6f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


# --------------------------------------------------------------------------
# 5. Prediction & Submission Function
# --------------------------------------------------------------------------
def predict_and_submit(model, test_loader, test_ids):
    """
    Generates predictions and saves submission file.
    """
    device = Config.DEVICE
    model.eval()
    model.to(device)

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # Convert predictions (0-6) back to target labels (1-7)
    # The dataset uses 1-based indexing for Cover_Type
    final_preds = np.array(predictions) + 1

    # Create submission DataFrame
    submission = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: final_preds})

    # Save
    save_path = Config.SUBMISSION_PATH
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
