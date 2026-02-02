import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import copy
from library.config import Config

# Set fixed seeds for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)


class CrossLayer(nn.Module):
    """
    Implements a single layer of the Cross Network (DCN-V2 vector form).
    Formula: x_{l+1} = x_0 * (x_l^T . w) + b + x_l
    """

    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.input_dim = input_dim
        # Weight and bias parameters
        self.w = nn.Parameter(torch.zeros(input_dim))
        self.b = nn.Parameter(torch.zeros(input_dim))

        # Initialize parameters
        nn.init.xavier_uniform_(self.w.unsqueeze(0))
        nn.init.zeros_(self.b)

    def forward(self, x0, xl):
        """
        Args:
            x0: Input to the network (Batch, Dim)
            xl: Output of the previous layer (Batch, Dim)
        Returns:
            x_{l+1}: Output of this layer (Batch, Dim)
        """
        # Compute dot product (xl^T . w) for each sample in batch
        # xl: (B, D), w: (D) -> xl * w -> (B, D) -> sum(dim=1) -> (B, 1)
        xw = (xl * self.w).sum(dim=1, keepdim=True)

        # x0 * scalar + b + xl
        out = x0 * xw + self.b + xl
        return out


class ResidualBlock(nn.Module):
    """
    Implements a Residual Block for the Deep Branch.
    Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> Skip Connection
    """

    def __init__(self, dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.layers(x)


class DCNResNet(nn.Module):
    """
    Parallel DCN-ResNet (Wide-Input) Architecture.
    Consists of a Deep Branch (ResNet) and a Cross Branch (DCN) processing
    the same dense input vector in parallel.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_res_blocks,
        num_cross_layers,
        num_classes,
        dropout_rate,
    ):
        super(DCNResNet, self).__init__()

        # Deep Branch
        # Projection from Input Dim to Hidden Dim
        self.deep_proj = nn.Linear(input_dim, hidden_dim)
        # Stack of Residual Blocks
        self.deep_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_res_blocks)]
        )

        # Cross Branch
        # Stack of Cross Layers
        self.cross_layers = nn.ModuleList(
            [CrossLayer(input_dim) for _ in range(num_cross_layers)]
        )

        # Combination Head
        # Concatenates output of Deep Branch (hidden_dim) and Cross Branch (input_dim)
        concat_dim = hidden_dim + input_dim
        self.final_head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # Deep Branch
        x_deep = self.deep_proj(x)
        x_deep = self.deep_blocks(x_deep)

        # Cross Branch
        x_cross = x
        for layer in self.cross_layers:
            x_cross = layer(x, x_cross)  # x0 is always the original input x

        # Concatenate
        out = torch.cat([x_deep, x_cross], dim=1)

        # Final Logits
        logits = self.final_head(out)
        return logits


def train_model(train_loader, val_loader, input_dim):
    """
    Trains the DCNResNet model with Early Stopping and ReduceLROnPlateau.
    """
    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = DCNResNet(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        num_cross_layers=Config.NUM_CROSS_LAYERS,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.FACTOR,
        patience=2,
        verbose=True,
        min_lr=Config.MIN_LR,
    )

    # Early Stopping Tracking
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        # Training Phase
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

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation Phase
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

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {epoch_loss:.6f} - Train Acc: {epoch_acc:.6f} - "
            f"Val Loss: {val_loss:.6f} - Val Acc: {val_acc:.6f}"
        )

        # Scheduler Step
        scheduler.step(val_acc)

        # Early Stopping Logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save best model immediately
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val Acc: {best_val_acc:.6f}")

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model


def predict(model, test_loader, test_ids):
    """
    Generates predictions for the test set and saves to CSV.
    """
    device = torch.device(Config.DEVICE)
    model.eval()
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs in test_loader:
            # inputs is a list containing one tensor if from TensorDataset
            if isinstance(inputs, list):
                inputs = inputs[0]
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Map 0-6 back to 1-7
            predicted = predicted.cpu().numpy() + 1
            predictions.extend(predicted)

    # Create Submission DataFrame
    df_submission = pd.DataFrame(
        {Config.ID_COL: test_ids, Config.TARGET_COL: predictions}
    )

    # Ensure ID is int
    df_submission[Config.ID_COL] = df_submission[Config.ID_COL].astype(int)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
