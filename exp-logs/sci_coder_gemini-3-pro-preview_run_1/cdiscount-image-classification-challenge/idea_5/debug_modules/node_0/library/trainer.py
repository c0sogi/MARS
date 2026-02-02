import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import GradScaler, autocast

# Import from provided libraries
import library.config as config
import library.dataset as dataset
import library.model as model_lib
import library.utils as utils


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss Functions
        # Level 3 (Fine) gets label smoothing
        self.criterion_l3 = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        # Auxiliary heads (L1, L2) use standard CE
        self.criterion_aux = nn.CrossEntropyLoss()

        self.loss_weights = config.LOSS_WEIGHTS

        # Optimizer: SGD with Momentum
        # We use the learning rate from config, which is the base for OneCycleLR
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True,
        )

        # Scheduler: OneCycleLR
        # Steps per epoch is len(train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            epochs=config.NUM_EPOCHS,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,  # Warmup for 30% of training
            div_factor=25.0,
            final_div_factor=1000.0,
        )

        # Mixed Precision Scaler
        self.scaler = GradScaler()

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0
        correct_l3 = 0
        total_samples = 0

        start_time = time.time()

        for batch_idx, (images, l1_targets, l2_targets, l3_targets) in enumerate(
            self.train_loader
        ):
            # Move data to device
            images = images.to(self.device, non_blocking=True)
            l1_targets = l1_targets.to(self.device, non_blocking=True)
            l2_targets = l2_targets.to(self.device, non_blocking=True)
            l3_targets = l3_targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast():
                logits_l1, logits_l2, logits_l3 = self.model(images)

                loss_l1 = self.criterion_aux(logits_l1, l1_targets)
                loss_l2 = self.criterion_aux(logits_l2, l2_targets)
                loss_l3 = self.criterion_l3(logits_l3, l3_targets)

                # Weighted Sum
                total_loss = (
                    self.loss_weights["level3"] * loss_l3
                    + self.loss_weights["level2"] * loss_l2
                    + self.loss_weights["level1"] * loss_l1
                )

            # Backward & Step
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # Metrics
            running_loss += total_loss.item() * images.size(0)

            _, preds_l3 = torch.max(logits_l3, 1)
            correct_l3 += (preds_l3 == l3_targets).sum().item()
            total_samples += images.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_l3 / total_samples
        duration = time.time() - start_time

        print(
            f"Epoch {epoch_idx+1}/{config.NUM_EPOCHS} [Train] "
            f"Loss: {epoch_loss} | L3 Acc: {epoch_acc} | Time: {duration:.2f}s"
        )

        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct_l3 = 0
        total_samples = 0

        with torch.no_grad():
            for images, l1_targets, l2_targets, l3_targets in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                l1_targets = l1_targets.to(self.device, non_blocking=True)
                l2_targets = l2_targets.to(self.device, non_blocking=True)
                l3_targets = l3_targets.to(self.device, non_blocking=True)

                # We don't strictly need autocast for eval, but it can speed it up
                with autocast():
                    logits_l1, logits_l2, logits_l3 = self.model(images)

                    loss_l1 = self.criterion_aux(logits_l1, l1_targets)
                    loss_l2 = self.criterion_aux(logits_l2, l2_targets)
                    loss_l3 = self.criterion_l3(logits_l3, l3_targets)

                    total_loss = (
                        self.loss_weights["level3"] * loss_l3
                        + self.loss_weights["level2"] * loss_l2
                        + self.loss_weights["level1"] * loss_l1
                    )

                running_loss += total_loss.item() * images.size(0)

                _, preds_l3 = torch.max(logits_l3, 1)
                correct_l3 += (preds_l3 == l3_targets).sum().item()
                total_samples += images.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_l3 / total_samples

        print(f"Epoch [Val] Loss: {epoch_loss} | L3 Acc: {epoch_acc}")

        return epoch_loss, epoch_acc

    def fit(self):
        best_acc = 0.0
        save_path = os.path.join(config.WORKING_DIR, "best_model.pth")

        print(f"Starting training on device: {self.device}")

        for epoch in range(config.NUM_EPOCHS):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()

            # Checkpoint
            if val_acc > best_acc:
                print(
                    f"Validation accuracy improved from {best_acc} to {val_acc}. Saving model..."
                )
                best_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
            else:
                print(f"Validation accuracy did not improve (Best: {best_acc}).")

        print(f"Training complete. Best Validation L3 Accuracy: {best_acc}")
        return best_acc


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed) # Not imported, but usually good practice
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enable benchmark for speed on fixed size


def run_training(debug=False, subset_size=2000, epochs=None):
    """
    Main entry point to run the training pipeline.

    Args:
        debug (bool): If True, uses a small subset of data.
        subset_size (int): Number of samples for debug mode.
        epochs (int, optional): Override config.NUM_EPOCHS.
    """
    set_seed(config.SEED)

    # Override epochs if provided
    if epochs is not None:
        config.NUM_EPOCHS = epochs

    device = torch.device(config.DEVICE)

    # 1. Get DataLoaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, mapper = dataset.get_dataloaders(
        debug=debug,
        subset_size=subset_size,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    stats = mapper.get_num_classes()
    print(
        f"Classes: L1={stats['num_classes_l1']}, L2={stats['num_classes_l2']}, L3={stats['num_classes_l3']}"
    )

    # 2. Initialize Model
    print("Initializing Model...")
    model = model_lib.DeepSupervisedResNet50(
        num_classes_l1=stats["num_classes_l1"],
        num_classes_l2=stats["num_classes_l2"],
        num_classes_l3=stats["num_classes_l3"],
        pretrained=True,
    )
    model.to(device)

    # 3. Setup Trainer and Fit
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit()

    return trainer
