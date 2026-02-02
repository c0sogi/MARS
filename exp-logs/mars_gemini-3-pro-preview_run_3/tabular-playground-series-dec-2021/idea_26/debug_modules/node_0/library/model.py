import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from library.config import Config


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Formula: x_{l+1} = x_0 * (x_l^T w) + b + x_l

    This layer captures explicit feature interactions by scaling the original input x0
    based on a learned projection of the current layer x_l.
    """

    def __init__(self, input_dim):
        super().__init__()
        # Parameter w: (input_dim,) - Learned projection vector
        self.w = nn.Parameter(torch.randn(input_dim))
        # Parameter b: (input_dim,) - Bias term
        self.b = nn.Parameter(torch.zeros(input_dim))

        # Initialize w with Xavier Uniform to ensure stable gradients at start
        nn.init.xavier_uniform_(self.w.unsqueeze(0))

    def forward(self, x, x0):
        """
        Args:
            x: Input tensor at current layer l (Batch, Input_Dim)
            x0: Original input tensor (Batch, Input_Dim)
        Returns:
            Tensor of shape (Batch, Input_Dim)
        """
        # Compute scalar interaction score: x_l^T w
        # x (B, D) * w (D,) -> (B, D) -> sum(dim=1) -> (B, 1)
        score = torch.sum(x * self.w, dim=1, keepdim=True)

        # Apply mixing: x0 * score + b + x
        # (B, D) * (B, 1) + (D,) + (B, D) -> (B, D)
        out = x0 * score + self.b + x
        return out


class PreActResBlock(nn.Module):
    """
    Pre-Activation Residual Block.
    Structure: BN -> ReLU -> Dropout -> Linear
    Residual: x + block(x)

    This structure creates a clean identity path for gradients, facilitating
    the training of deeper networks.
    """

    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

        # He initialization for ReLU
        nn.init.kaiming_normal_(self.linear.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        # Pre-activation: BN -> ReLU
        out = self.bn(x)
        out = F.relu(out)
        # Regularization
        out = self.dropout(out)
        # Weight layer
        out = self.linear(out)
        # Residual connection
        return x + out


class DeepPreActDCNResNet(nn.Module):
    """
    Deep Pre-Activation Parallel DCN-ResNet.

    Architecture:
    1. Input: Dense concatenated vector.
    2. Branch 1 (DCN): Stack of VectorCrossLayers for explicit feature interactions.
    3. Branch 2 (ResNet): Deep Pre-Activation ResNet for high-capacity representation learning.
    4. Head: Concatenation of branches -> Linear Classifier.
    """

    def __init__(self, input_dim=None):
        super().__init__()

        # Determine input dimension
        if input_dim is None:
            if hasattr(Config, "INPUT_DIM"):
                self.input_dim = Config.INPUT_DIM
            else:
                raise ValueError(
                    "input_dim must be provided explicitly or via Config.INPUT_DIM"
                )
        else:
            self.input_dim = input_dim

        self.hidden_dim = Config.HIDDEN_DIM
        self.num_blocks = Config.NUM_RESNET_BLOCKS
        self.dropout_rate = Config.DROPOUT_RATE
        self.num_classes = Config.NUM_CLASSES

        # --- Branch 1: DCN (Explicit Interactions) ---
        # We use a stack of cross layers. The depth matches the ResNet blocks for symmetry.
        self.dcn_layers = nn.ModuleList(
            [VectorCrossLayer(self.input_dim) for _ in range(self.num_blocks)]
        )

        # --- Branch 2: ResNet (Deep Representation) ---
        # Stem: Project input to hidden dimension
        self.resnet_stem = nn.Linear(self.input_dim, self.hidden_dim)
        nn.init.kaiming_normal_(
            self.resnet_stem.weight, mode="fan_in", nonlinearity="linear"
        )

        # Backbone: Stack of Pre-Activation Blocks
        self.resnet_blocks = nn.ModuleList(
            [
                PreActResBlock(self.hidden_dim, self.dropout_rate)
                for _ in range(self.num_blocks)
            ]
        )

        # Final Normalization/Activation for ResNet branch
        # Ensures features are well-scaled before concatenation
        self.resnet_final_bn = nn.BatchNorm1d(self.hidden_dim)

        # --- Combination Head ---
        # Concatenate DCN output (input_dim) and ResNet output (hidden_dim)
        concat_dim = self.input_dim + self.hidden_dim
        self.head = nn.Linear(concat_dim, self.num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        # --- Branch 1: DCN Forward ---
        x_dcn = x
        for layer in self.dcn_layers:
            # Cross layer requires current input and original input x0
            x_dcn = layer(x_dcn, x)

        # --- Branch 2: ResNet Forward ---
        x_res = self.resnet_stem(x)
        for block in self.resnet_blocks:
            x_res = block(x_res)

        # Apply final pre-activation style norm/act to the residual branch output
        x_res = self.resnet_final_bn(x_res)
        x_res = F.relu(x_res)

        # --- Combination ---
        combined = torch.cat([x_dcn, x_res], dim=1)
        logits = self.head(combined)

        return logits


def train_model(model, train_loader, val_loader, device=None, epochs=Config.EPOCHS):
    """
    Trains the DeepPreActDCNResNet model with AdamW, ReduceLROnPlateau, and Early Stopping.
    """
    if device is None:
        device = torch.device(Config.DEVICE)

    model = model.to(device)

    # Optimizer: AdamW (Decoupled Weight Decay)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=Config.SCHEDULER_MODE,
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    criterion = nn.CrossEntropyLoss()

    # Early Stopping State
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        avg_train_loss = train_loss / total_train
        train_acc = correct_train / total_train

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        avg_val_loss = val_loss / total_val
        val_acc = correct_val / total_val

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss} | Train Acc: {train_acc} | Val Loss: {avg_val_loss} | Val Acc: {val_acc}"
        )

        # --- Scheduler Step ---
        # Monitor Validation Accuracy (max mode)
        scheduler.step(val_acc)

        # --- Early Stopping ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save checkpoint immediately
            torch.save(best_model_state, Config.MODEL_CHECKPOINT_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. Best Val Acc: {best_val_acc}"
                )
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored best model weights.")

    return model


def generate_submission(
    model, test_loader, test_ids, device=None, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    if device is None:
        device = torch.device(Config.DEVICE)

    model = model.to(device)
    model.eval()

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # Convert back to 1-based indexing (0-6 -> 1-7)
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())

    # Ensure lengths match
    if len(test_ids) != len(predictions):
        print(
            f"Warning: Number of IDs ({len(test_ids)}) does not match predictions ({len(predictions)})"
        )

    # Create submission DataFrame
    submission_df = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
