import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import roc_auc_score
import numpy as np

from library.config import Config
from library.dataset import get_dataloaders
from library.model import CustomRepVGG
from library.utils import mixup_data, mixup_criterion, update_bn, MetricMonitor


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()

        # SWA components
        self.swa_model = AveragedModel(model)
        self.swa_start_epoch = Config.SWA_START_EPOCH

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Schedulers
        # Phase 1: Cosine Annealing until SWA start
        # We set T_max to swa_start_epoch so it decays fully before SWA takes over
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.swa_start_epoch, eta_min=1e-6
        )

        # Phase 2: SWA LR (Constant or Cyclic)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=Config.SWA_LR)

        self.best_auc = 0.0

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        metric_monitor = MetricMonitor()

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device).float().view(-1, 1)

            # Mixup Regularization
            images, targets_a, targets_b, lam = mixup_data(
                images,
                labels,
                alpha=Config.MIXUP_ALPHA,
                use_cuda=(self.device != "cpu"),
            )

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            self.optimizer.step()

            metric_monitor.update("Loss", loss.item())

        return metric_monitor

    def validate(self, model_to_validate):
        model_to_validate.eval()
        metric_monitor = MetricMonitor()

        preds = []
        targets = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float().view(-1, 1)

                outputs = model_to_validate(images)
                probs = torch.sigmoid(outputs)

                preds.append(probs.cpu().numpy())
                targets.append(labels.cpu().numpy())

                loss = self.criterion(outputs, labels)
                metric_monitor.update("Loss", loss.item())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        try:
            auc = roc_auc_score(targets, preds)
        except ValueError:
            # Handle edge case where batch might have only one class
            auc = 0.5

        metric_monitor.update("AUC", auc)
        return metric_monitor, auc

    def fit(self, epochs):
        print(f"Starting training for {epochs} epochs...")
        print(f"SWA will start at epoch {self.swa_start_epoch}")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_one_epoch(epoch)

            # Scheduler & SWA Logic
            if epoch > self.swa_start_epoch:
                # Phase 2: Update SWA model and use SWA scheduler
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                lr = self.swa_scheduler.get_last_lr()[0]
                is_swa_phase = True
            else:
                # Phase 1: Standard scheduler
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
                is_swa_phase = False

            # Validate Base Model (to monitor convergence)
            val_metrics, auc = self.validate(self.model)

            # Save Best Base Model
            if auc > self.best_auc:
                self.best_auc = auc
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"Epoch {epoch}: New Best Base AUC: {auc}")

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch}/{epochs} | LR: {lr:.6f} | "
                f"Train: {train_metrics} | Val: {val_metrics} | "
                f"Time: {epoch_time:.2f}s | SWA: {is_swa_phase}"
            )

        total_time = time.time() - start_time
        print(f"Training finished in {total_time:.2f}s")

        # Finalize SWA Model
        if epochs > self.swa_start_epoch:
            print("Finalizing SWA Model (Updating BN statistics)...")
            # Update BN stats for the averaged model using training data
            update_bn(self.train_loader, self.swa_model, device=self.device)

            print("Validating SWA Model...")
            swa_metrics, swa_auc = self.validate(self.swa_model)
            print(f"SWA Model Results: {swa_metrics}")

            # Save SWA model
            torch.save(self.swa_model.module.state_dict(), Config.FINAL_SWA_MODEL_PATH)
            print(f"Saved SWA model to {Config.FINAL_SWA_MODEL_PATH}")

            return self.swa_model.module
        else:
            print("SWA was not triggered. Returning best base model.")
            # Load best base model
            self.model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
            return self.model


def run_training():
    """
    Main entry point for training the model.
    """
    set_seed(Config.SEED)

    # Get DataLoaders (uses caching internally via library.dataset)
    train_loader, val_loader = get_dataloaders()

    # Initialize Model
    # We initialize with deploy=False for training
    model = CustomRepVGG(num_classes=Config.NUM_CLASSES, deploy=False)
    model = model.to(Config.DEVICE)

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, Config.DEVICE)

    # Run Training
    final_model = trainer.fit(Config.EPOCHS)

    return final_model
