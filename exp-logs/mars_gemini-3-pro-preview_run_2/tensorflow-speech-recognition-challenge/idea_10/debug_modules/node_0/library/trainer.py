import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import AverageMeter, save_checkpoint, get_device, set_seed
from library.model import EfficientNetV2Audio
from library.dataset import get_dataloaders


class ModelEMA:
    """
    Model Exponential Moving Average (EMA).
    Maintains a moving average of model parameters for better generalization.
    """

    def __init__(self, model, decay=Config.EMA_DECAY, device=None):
        self.decay = decay
        self.model = model
        self.device = device if device else get_device()

        # Create a shadow copy of the model
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.ema.to(self.device)

        # Disable gradients for the shadow model
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA parameters based on the current model parameters.
        Formula: ema_param = decay * ema_param + (1 - decay) * current_param
        """
        with torch.no_grad():
            # Update parameters
            msd = model.state_dict()
            esd = self.ema.state_dict()

            for name, param in msd.items():
                if name in esd:
                    if param.dtype.is_floating_point:
                        esd[name].copy_(
                            esd[name] * self.decay + param * (1.0 - self.decay)
                        )
                    else:
                        # Copy non-floating point parameters (e.g. integer buffers) directly
                        esd[name].copy_(param)


class Trainer:
    """
    Trainer class to manage the training, validation, and optimization lifecycle.
    """

    def __init__(self):
        self.device = get_device()
        set_seed(Config.SEED)

        # 1. Initialize Model
        print(f"Initializing model: {Config.MODEL_NAME}")
        self.model = EfficientNetV2Audio(
            num_classes=Config.NUM_CLASSES, pretrained=True
        )
        self.model.to(self.device)

        # 2. Initialize EMA
        self.ema = ModelEMA(self.model, decay=Config.EMA_DECAY, device=self.device)

        # 3. Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        # 4. Loss Function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        # 5. Data Loaders
        print("Loading data...")
        self.train_loader, self.val_loader, _ = get_dataloaders(load_cached_data=True)

        # 6. Training State
        self.start_epoch = 0
        self.best_acc = 0.0
        self.patience_counter = 0

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()

        losses = AverageMeter()
        accuracies = AverageMeter()

        start_time = time.time()

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update EMA
            self.ema.update(self.model)

            # Metrics
            acc = (logits.argmax(dim=1) == labels).float().mean().item()
            losses.update(loss.item(), images.size(0))
            accuracies.update(acc, images.size(0))

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] Train Loss: {losses.avg} | Train Acc: {accuracies.avg} | Time: {epoch_time}s"
        )
        return losses.avg, accuracies.avg

    def validate(self):
        """
        Runs validation using the EMA model.
        """
        # Use EMA model for validation/inference
        model_to_eval = self.ema.ema
        model_to_eval.eval()

        losses = AverageMeter()
        accuracies = AverageMeter()

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = model_to_eval(images)
                loss = self.criterion(logits, labels)

                acc = (logits.argmax(dim=1) == labels).float().mean().item()
                losses.update(loss.item(), images.size(0))
                accuracies.update(acc, images.size(0))

        print(f"Validation Loss: {losses.avg} | Validation Acc: {accuracies.avg}")
        return losses.avg, accuracies.avg

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print("Starting training...")

        for epoch in range(self.start_epoch, Config.NUM_EPOCHS):
            # Train
            train_loss, train_acc = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate()

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Current LR: {current_lr}")

            # Checkpointing (Save EMA weights as they are used for inference)
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                self.patience_counter = 0
                print(f"New best model found! Accuracy: {self.best_acc}")
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{Config.PATIENCE}"
                )

            # Save state
            # We save the EMA model state_dict as the primary 'state_dict' for easy loading in inference
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.ema.ema.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "best_acc": self.best_acc,
                },
                is_best,
            )

            # Early Stopping
            if self.patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training finished. Best Validation Accuracy: {self.best_acc}")
