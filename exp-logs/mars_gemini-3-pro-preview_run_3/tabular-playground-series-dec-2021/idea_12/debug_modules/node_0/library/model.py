import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import copy
from library.config import Config
from library.utils import AverageMeter, calculate_accuracy, EarlyStopping


class VectorCrossLayer(nn.Module):
    """
    Vector-based Cross Layer for Deep & Cross Network (DCN-V2).
    Implements the formula: x_{l+1} = x_0 * (x_l^T w) + b + x_l
    Uses dot-product mixing to ensure proper interaction.
    """

    def __init__(self, input_dim):
        super(VectorCrossLayer, self).__init__()
        self.input_dim = input_dim
        # Weight w: (D, 1)
        self.weight = nn.Parameter(torch.randn(input_dim, 1))
        # Bias b: (D, )
        self.bias = nn.Parameter(torch.zeros(input_dim))

        # Initialization
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x0, xl):
        """
        Args:
            x0: Original input features (Batch, D)
            xl: Output from previous cross layer (Batch, D)
        Returns:
            x_next: (Batch, D)
        """
        # Compute scalar score per sample: (Batch, D) @ (D, 1) -> (Batch, 1)
        score = torch.mm(xl, self.weight)

        # Mix with x0: (Batch, D) * (Batch, 1) -> (Batch, D)
        # Broadcasting ensures x0 is scaled by the interaction score
        mixed = x0 * score

        # Add bias and residual
        x_next = mixed + self.bias + xl
        return x_next


class ResNeXtBlock(nn.Module):
    """
    ResNeXt Block using 1D Grouped Convolutions to simulate Grouped Linear Transformations.
    Structure:
    Input -> GroupedConv1D -> BN -> ReLU -> GroupedConv1D -> BN -> Residual -> ReLU
    """

    def __init__(self, dim, groups=32):
        super(ResNeXtBlock, self).__init__()

        # Ensure dim is divisible by groups
        if dim % groups != 0:
            raise ValueError(f"Dimension {dim} must be divisible by groups {groups}")

        self.conv1 = nn.Conv1d(dim, dim, kernel_size=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Dim, 1)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ParallelDCNResNeXt(nn.Module):
    """
    Hybrid architecture with parallel DCN and ResNeXt branches.
    """

    def __init__(self, input_dim, num_classes):
        super(ParallelDCNResNeXt, self).__init__()

        # ==========================
        # Branch 1: Vector-DCN
        # ==========================
        self.num_cross_layers = Config.DCN_LAYERS
        self.dcn_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(self.num_cross_layers)]
        )

        # ==========================
        # Branch 2: ResNeXt Backbone
        # ==========================
        self.resnext_hidden = Config.RESNEXT_HIDDEN_DIM
        self.groups = Config.RESNEXT_GROUPS
        self.resnext_layers = Config.RESNEXT_LAYERS

        # Projection to hidden dimension
        self.resnext_projection = nn.Linear(input_dim, self.resnext_hidden)
        self.resnext_bn_in = nn.BatchNorm1d(self.resnext_hidden)
        self.resnext_relu_in = nn.ReLU(inplace=True)

        # Stack of ResNeXt Blocks
        blocks = []
        for _ in range(self.resnext_layers):
            blocks.append(ResNeXtBlock(self.resnext_hidden, groups=self.groups))
        self.resnext_backbone = nn.Sequential(*blocks)

        # ==========================
        # Combination Head
        # ==========================
        concat_dim = input_dim + self.resnext_hidden

        self.dropout = nn.Dropout(Config.DROPOUT)
        self.classifier = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # x shape: (Batch, input_dim)

        # --- Branch 1: DCN ---
        x_cross = x
        for layer in self.dcn_layers:
            x_cross = layer(x, x_cross)
        # x_cross shape: (Batch, input_dim)

        # --- Branch 2: ResNeXt ---
        x_res = self.resnext_projection(x)
        x_res = self.resnext_bn_in(x_res)
        x_res = self.resnext_relu_in(x_res)

        # Reshape for Conv1d: (Batch, Hidden) -> (Batch, Hidden, 1)
        x_res = x_res.unsqueeze(2)

        x_res = self.resnext_backbone(x_res)

        # Flatten: (Batch, Hidden, 1) -> (Batch, Hidden)
        x_res = x_res.squeeze(2)

        # --- Combination ---
        combined = torch.cat([x_cross, x_res], dim=1)
        combined = self.dropout(combined)

        logits = self.classifier(combined)

        return logits


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(output, target)
        losses.update(loss.item(), data.size(0))
        accuracies.update(acc, data.size(0))

    return losses.avg, accuracies.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            acc = calculate_accuracy(output, target)
            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))

    return losses.avg, accuracies.avg


def train_model(
    train_loader,
    val_loader,
    input_dim,
    num_classes=Config.NUM_CLASSES,
    epochs=Config.EPOCHS,
    device=Config.DEVICE,
):
    """
    Main training loop with Early Stopping and Cosine Annealing.
    """
    print(
        f"Initializing ParallelDCNResNeXt with Input Dim: {input_dim}, Classes: {num_classes}"
    )
    model = ParallelDCNResNeXt(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    # Early Stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE, mode="max")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Check Early Stopping
        early_stopping(val_acc, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load best weights
    print("Loading best model weights...")
    early_stopping.load_best_weights(model)

    return model


def generate_submission(
    model, test_loader, device=Config.DEVICE, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    predictions = []
    ids_list = []

    print("Generating predictions...")
    with torch.no_grad():
        for data, ids in test_loader:
            data = data.to(device)
            output = model(data)

            # Get predicted class index (0-6)
            preds = torch.argmax(output, dim=1).cpu().numpy()

            # Convert back to original class labels (1-7)
            preds = preds + 1

            predictions.extend(preds)
            ids_list.extend(ids.numpy())

    # Create DataFrame
    df_sub = pd.DataFrame({"Id": ids_list, "Cover_Type": predictions})

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
