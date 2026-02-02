import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import copy
import time

from library.config import Config
from library.utils import get_logger, seed_everything, AverageMeter
from library.data_loader import get_dataloaders

logger = get_logger()

# ==========================================
# Model Architecture
# ==========================================


class FactorizedCrossLayer(nn.Module):
    """
    Implements a single layer of the Factorized Deep & Cross Network.
    Formula: x_{l+1} = x_0 * (U * (V^T * x_l) + b) + x_l
    Where W = U * V^T is a low-rank decomposition of the weight matrix.
    """

    def __init__(self, input_dim, rank):
        super(FactorizedCrossLayer, self).__init__()
        self.input_dim = input_dim
        self.rank = rank

        # V: projects input_dim -> rank
        # We use bias=False for the first projection to strictly model V^T * x
        self.V = nn.Linear(input_dim, rank, bias=False)

        # U: projects rank -> input_dim
        # We include the bias 'b' here: U * (V^T x) + b
        self.U = nn.Linear(rank, input_dim, bias=True)

        # Initialization
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.U.weight)

    def forward(self, x_0, x_l):
        # x_l: [batch_size, input_dim]
        # x_0: [batch_size, input_dim]

        # Project down: V^T * x_l -> [batch, rank]
        proj_low = self.V(x_l)

        # Project up: U * proj_low + b -> [batch, input_dim]
        proj_high = self.U(proj_low)

        # Element-wise multiply with x_0 (Broadcasting happens automatically)
        interaction = x_0 * proj_high

        # Residual connection
        return interaction + x_l


class ResNetBlock(nn.Module):
    """
    Standard Residual Block for Tabular Data.
    Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Residual
    """

    def __init__(self, hidden_dim, dropout):
        super(ResNetBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        # Post-activation residual connection
        out = F.relu(out + residual)
        return out


class ParallelFactorizedDCNResNet(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        dcn_rank=Config.DCN_RANK,
        hidden_dim=Config.HIDDEN_DIM,
        resnet_blocks=Config.RESNET_BLOCKS,
        dropout=Config.DROPOUT,
    ):
        super(ParallelFactorizedDCNResNet, self).__init__()

        # Branch 1: Factorized DCN
        # Stack of cross layers. Using 3 layers as a robust default.
        self.num_cross_layers = 3
        self.cross_layers = nn.ModuleList(
            [
                FactorizedCrossLayer(input_dim, dcn_rank)
                for _ in range(self.num_cross_layers)
            ]
        )

        # Branch 2: Wide ResNet Backbone
        self.resnet_input_proj = nn.Linear(input_dim, hidden_dim)
        self.resnet_input_bn = nn.BatchNorm1d(hidden_dim)
        self.resnet_blocks = nn.Sequential(
            *[ResNetBlock(hidden_dim, dropout) for _ in range(resnet_blocks)]
        )

        # Combination Head
        # Concatenate DCN output (input_dim) and ResNet output (hidden_dim)
        concat_dim = input_dim + hidden_dim
        self.head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # x: [batch, input_dim]

        # Branch 1: DCN
        x_dcn = x
        for layer in self.cross_layers:
            x_dcn = layer(x, x_dcn)  # Pass x_0 (original input) and x_l (current)

        # Branch 2: ResNet
        x_res = self.resnet_input_proj(x)
        x_res = self.resnet_input_bn(x_res)
        x_res = F.relu(x_res)
        x_res = self.resnet_blocks(x_res)

        # Combine
        x_concat = torch.cat([x_dcn, x_res], dim=1)
        logits = self.head(x_concat)

        return logits


# ==========================================
# Training & Evaluation Logic
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        # Metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch_y).float().mean()

        loss_meter.update(loss.item(), batch_X.size(0))
        acc_meter.update(acc.item(), batch_X.size(0))

    return loss_meter.avg, acc_meter.avg


def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            preds = torch.argmax(logits, dim=1)
            acc = (preds == batch_y).float().mean()

            loss_meter.update(loss.item(), batch_X.size(0))
            acc_meter.update(acc.item(), batch_X.size(0))

    return loss_meter.avg, acc_meter.avg


def predict_test(model, loader, device):
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch_X, batch_ids in loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            preds = torch.argmax(logits, dim=1)

            # Convert back to 1-7 range (preds are 0-6)
            preds = preds + 1

            all_preds.extend(preds.cpu().numpy())
            all_ids.extend(batch_ids.numpy())

    return all_ids, all_preds


def run_experiment():
    """
    Main execution function to train the model and generate submission.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # 1. Data
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Determine input dim from data
    dummy_X, _ = next(iter(train_loader))
    input_dim = dummy_X.shape[1]

    # Determine num classes (Cover_Type 1-7 -> 7 classes)
    num_classes = 7

    logger.info(f"Input Dimension: {input_dim}")
    logger.info(f"Num Classes: {num_classes}")

    # 2. Model
    model = ParallelFactorizedDCNResNet(input_dim, num_classes).to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.SCHEDULER_ETA_MIN
    )

    # 4. Training Loop
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print metrics with full precision
        logger.info(
            f"Epoch {epoch}/{Config.EPOCHS} | LR: {current_lr:.8f} | "
            f"Train Loss: {train_loss:.8f} | Train Acc: {train_acc:.8f} | "
            f"Val Loss: {val_loss:.8f} | Val Acc: {val_acc:.8f}"
        )

        # Checkpointing & Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time
    logger.info(
        f"Training finished in {total_time:.2f}s. Best Val Acc: {best_val_acc:.8f}"
    )

    # 5. Prediction
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    logger.info("Generating predictions on Test set...")
    ids, preds = predict_test(model, test_loader, device)

    # 6. Submission
    logger.info(f"Saving submission to {Config.SUBMISSION_FILE}...")
    df_sub = pd.DataFrame({"Id": ids, "Cover_Type": preds})

    # Ensure integer types
    df_sub["Id"] = df_sub["Id"].astype(int)
    df_sub["Cover_Type"] = df_sub["Cover_Type"].astype(int)

    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    logger.info("Submission saved successfully.")
