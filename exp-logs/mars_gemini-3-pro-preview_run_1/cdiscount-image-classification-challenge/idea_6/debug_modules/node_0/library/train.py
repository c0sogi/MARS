import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from library.config import Config
from library.utils import HierarchyManager
from library.dataset import BSONDataset
from library.model import MultiLevelResNet


# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class HierarchicalLoss(nn.Module):
    """
    Computes the weighted sum of losses for the three hierarchical heads.
    Dynamically retrieves parent labels for auxiliary supervision.
    """

    def __init__(self, hierarchy_manager, device, label_smoothing=0.0):
        super(HierarchicalLoss, self).__init__()
        self.hierarchy_manager = hierarchy_manager
        self.device = device

        # Loss functions
        # Fine-grained target gets label smoothing
        self.criterion_l3 = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # Auxiliary targets use standard CE
        self.criterion_l2 = nn.CrossEntropyLoss()
        self.criterion_l1 = nn.CrossEntropyLoss()

        # Weights
        self.w_l3 = Config.LOSS_WEIGHT_FINE
        self.w_l2 = Config.LOSS_WEIGHT_MID
        self.w_l1 = Config.LOSS_WEIGHT_COARSE

    def forward(self, preds, targets_l3):
        """
        Args:
            preds: Tuple (logits_l3, logits_l2, logits_l1)
            targets_l3: Tensor of shape (B,) with Level 3 class indices
        """
        logits_l3, logits_l2, logits_l1 = preds

        # Get auxiliary targets
        # targets_l3 are already on device, get_auxiliary_labels handles mapping
        targets_l1, targets_l2 = self.hierarchy_manager.get_auxiliary_labels(targets_l3)

        # Compute losses
        loss_l3 = self.criterion_l3(logits_l3, targets_l3)
        loss_l2 = self.criterion_l2(logits_l2, targets_l2)
        loss_l1 = self.criterion_l1(logits_l1, targets_l1)

        # Weighted sum
        total_loss = (
            (self.w_l3 * loss_l3) + (self.w_l2 * loss_l2) + (self.w_l1 * loss_l1)
        )

        return total_loss, (loss_l3.item(), loss_l2.item(), loss_l1.item())


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, device, epoch
):
    model.train()
    running_loss = 0.0
    correct_l3 = 0
    total_samples = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        # Unpack batch
        images = batch["images"].to(device, non_blocking=True)  # (B, 4, 3, H, W)
        mask = batch["mask"].to(device, non_blocking=True)  # (B, 4)
        targets = batch["target"].to(device, non_blocking=True)  # (B,)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with autocast():
            # Model returns tuple: (logits_l3, logits_l2, logits_l1)
            preds = model(images, mask)
            loss, component_losses = criterion(preds, targets)

        # Scaled Backward Pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Step Scheduler (OneCycleLR steps per batch)
        scheduler.step()

        # Metrics
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size

        # Calculate accuracy for target task (L3)
        logits_l3 = preds[0]
        _, predicted = torch.max(logits_l3, 1)
        correct_l3 += (predicted == targets).sum().item()
        total_samples += batch_size

    avg_loss = running_loss / total_samples
    accuracy = correct_l3 / total_samples
    duration = time.time() - start_time

    print(
        f"Epoch {epoch} [Train]: Loss: {avg_loss:.4f}, Acc L3: {accuracy:.4f}, Time: {duration:.2f}s"
    )
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_l3 = 0
    total_samples = 0

    start_time = time.time()

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            # Forward pass (no autocast needed for eval usually, but safe to use)
            preds = model(images, mask)
            loss, _ = criterion(preds, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size

            logits_l3 = preds[0]
            _, predicted = torch.max(logits_l3, 1)
            correct_l3 += (predicted == targets).sum().item()
            total_samples += batch_size

    avg_loss = running_loss / total_samples
    accuracy = correct_l3 / total_samples
    duration = time.time() - start_time

    print(
        f"Epoch Val [Valid]: Loss: {avg_loss:.4f}, Acc L3: {accuracy:.6f}, Time: {duration:.2f}s"
    )
    return avg_loss, accuracy


def run_training(limit_train_size=None, limit_val_size=None):
    """
    Main driver function to run the training pipeline.
    """
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Initialize Hierarchy Manager
    print("Loading Hierarchy Mappings...")
    hierarchy_manager = HierarchyManager(load_cached_data=True)

    # 2. Datasets and Loaders
    print("Initializing Datasets...")
    train_dataset = BSONDataset(
        metadata_path=Config.TRAIN_METADATA,
        bson_path=Config.TRAIN_BSON,
        split="train",
        limit_size=limit_train_size,
    )

    val_dataset = BSONDataset(
        metadata_path=Config.VAL_METADATA,
        bson_path=Config.TRAIN_BSON,
        split="val",
        limit_size=limit_val_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Model Setup
    print("Initializing Model...")
    model = MultiLevelResNet()
    model.to(device)

    # 4. Optimizer and Scheduler
    # Scale LR based on batch size (Linear Scaling Rule)
    # Base LR 1e-2 is good for BS=256, if BS=512 we might want 2e-2, etc.
    # We'll use the config value as the max_lr.
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.BASE_LR, weight_decay=Config.WEIGHT_DECAY
    )

    # One Cycle LR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.BASE_LR,
        epochs=Config.NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
    )

    # 5. Loss and Scaler
    criterion = HierarchicalLoss(
        hierarchy_manager=hierarchy_manager,
        device=device,
        label_smoothing=Config.LABEL_SMOOTHING,
    )
    scaler = GradScaler()

    # 6. Training Loop
    best_val_acc = 0.0
    patience = 3  # Not strictly needed for fixed epoch run, but good practice
    patience_counter = 0

    print("Starting Training...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Checkpoint
        if val_acc > best_val_acc:
            print(f"Validation Accuracy Improved: {best_val_acc:.6f} -> {val_acc:.6f}")
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation Accuracy did not improve. Best: {best_val_acc:.6f}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training Complete. Best Validation Accuracy: {best_val_acc:.6f}")
    print(f"Best model saved to: {Config.MODEL_CHECKPOINT}")
