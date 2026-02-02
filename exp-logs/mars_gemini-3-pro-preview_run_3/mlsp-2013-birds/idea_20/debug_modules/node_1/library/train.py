import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_robust_auc, AverageMeter, Logger
from library.dataset import (
    load_data,
    BirdDataset,
    get_transforms,
    mixup_data,
    mixup_criterion,
)
from library.models import get_model
from library.sam import SAM

try:
    from skmultilearn.model_selection import IterativeStratification

    HAS_SKMULTILEARN = True
except ImportError:
    HAS_SKMULTILEARN = False
    from sklearn.model_selection import KFold


class Trainer:
    """
    Manages the training and validation lifecycle for a single model.
    """

    def __init__(
        self,
        model,
        device,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        logger,
        fold,
        model_name,
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.logger = logger
        self.fold = fold
        self.model_name = model_name

        self.best_auc = 0.0
        self.patience_counter = 0
        self.best_epoch = 0

    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()

        # Determine if we are using SAM (requires closure)
        is_sam = isinstance(self.optimizer, SAM)

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            # Apply Mixup
            if Config.USE_MIXUP:
                images, targets_a, targets_b, lam = mixup_data(
                    images, labels, Config.MIXUP_ALPHA, use_cuda=True
                )
            else:
                targets_a, targets_b, lam = labels, labels, 1.0

            # Forward pass and loss calculation
            outputs = self.model(images)
            loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)

            # Optimization Step
            if is_sam:
                # 1. First step: Backward to compute gradients
                loss.backward()

                # 2. Define closure for SAM (re-evaluates loss at perturbed state)
                def closure():
                    self.optimizer.zero_grad()
                    output_closure = self.model(images)
                    loss_closure = mixup_criterion(
                        self.criterion, output_closure, targets_a, targets_b, lam
                    )
                    loss_closure.backward()
                    return loss_closure

                # 3. Step (Ascent -> Closure -> Descent)
                self.optimizer.step(closure)
                # Zero grads for next iteration (standard practice, though SAM handles internal zeroing)
                self.optimizer.zero_grad()
            else:
                # Standard Optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            losses.update(loss.item(), batch_size)

        return losses.avg

    def validate_epoch(self):
        self.model.eval()
        losses = AverageMeter()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_size = images.size(0)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                losses.update(loss.item(), batch_size)

                # Apply sigmoid for probabilities
                preds = torch.sigmoid(outputs)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        auc = calculate_robust_auc(all_targets, all_preds)
        return losses.avg, auc

    def fit(self):
        self.logger.log(f"Starting training for {self.model_name} - Fold {self.fold}")

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss, val_auc = self.validate_epoch()

            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = Config.LEARNING_RATE

            elapsed = time.time() - start_time

            self.logger.log(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Time: {elapsed:.1f}s | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc}"  # Full precision
            )

            # Checkpoint & Early Stopping
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_epoch = epoch
                self.patience_counter = 0

                save_path = os.path.join(
                    Config.CHECKPOINT_DIR,
                    f"{self.model_name}_fold_{self.fold}_best.pth",
                )
                torch.save(self.model.state_dict(), save_path)
                self.logger.log(f"  -> New Best AUC! Model saved to {save_path}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= Config.PATIENCE:
                self.logger.log(
                    f"Early stopping triggered at epoch {epoch}. Best AUC: {self.best_auc} at epoch {self.best_epoch}"
                )
                break

        return self.best_auc


def get_folds(images, labels, num_folds=5, seed=42):
    """
    Generates fold indices using Iterative Stratification or KFold.
    """
    # Create an index array
    X = np.zeros((len(labels), 1))  # Dummy X
    y = labels

    folds = []

    if HAS_SKMULTILEARN:
        k_fold = IterativeStratification(n_splits=num_folds, order=1)
        for train_idx, val_idx in k_fold.split(X, y):
            folds.append((train_idx, val_idx))
    else:
        print("skmultilearn not found. Falling back to KFold (random split).")
        k_fold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for train_idx, val_idx in k_fold.split(X):
            folds.append((train_idx, val_idx))

    return folds


def run_training(debug=False):
    """
    Main entry point for training.
    """
    seed_everything(Config.SEED)
    Config.setup()

    logger = Logger(os.path.join(Config.WORKING_DIR, "training_log.txt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}")

    # 1. Load Data
    # We combine train and val metadata to perform 5-fold CV
    train_imgs, train_lbls = load_data("train")
    val_imgs, val_lbls = load_data("val")

    # Concatenate
    all_images = np.concatenate([train_imgs, val_imgs], axis=0)
    all_labels = np.concatenate([train_lbls, val_lbls], axis=0)

    logger.log(f"Total dev samples: {len(all_images)}")

    if debug:
        logger.log("Debug mode: using subset of data")
        all_images = all_images[:50]
        all_labels = all_labels[:50]

    # 2. Prepare Folds
    folds = get_folds(
        all_images, all_labels, num_folds=Config.NUM_FOLDS, seed=Config.SEED
    )

    # 3. Iterate Architectures and Folds
    results = {}

    for model_name in Config.MODEL_ARCHITECTURES:
        model_scores = []
        logger.log(f"\n{'='*20} Training {model_name} {'='*20}")

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            logger.log(f"\n--- Fold {fold_idx} ---")

            # Split Data
            X_train, X_val = all_images[train_idx], all_images[val_idx]
            y_train, y_val = all_labels[train_idx], all_labels[val_idx]

            # Datasets
            train_dataset = BirdDataset(
                X_train, y_train, transform=get_transforms("train")
            )
            val_dataset = BirdDataset(X_val, y_val, transform=get_transforms("val"))

            # Dataloaders
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

            # Model
            model = get_model(model_name, pretrained=True)
            model = model.to(device)

            # Criterion
            # BCEWithLogitsLoss is standard for multi-label
            criterion = nn.BCEWithLogitsLoss()

            # Optimizer & Scheduler
            # Using SAM wrapping AdamW
            base_optimizer = torch.optim.AdamW
            optimizer = SAM(
                model.parameters(),
                base_optimizer=base_optimizer,
                rho=Config.SAM_RHO,
                adaptive=False,
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )

            scheduler = CosineAnnealingLR(
                optimizer.base_optimizer,
                T_max=Config.EPOCHS,
                eta_min=Config.MIN_LEARNING_RATE,
            )

            # Trainer
            trainer = Trainer(
                model=model,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                logger=logger,
                fold=fold_idx,
                model_name=model_name,
            )

            # Run Training
            best_score = trainer.fit()
            model_scores.append(best_score)

            # Cleanup
            del model, optimizer, scheduler, trainer, train_loader, val_loader
            torch.cuda.empty_cache()

        avg_score = np.mean(model_scores)
        results[model_name] = avg_score
        logger.log(f"Finished {model_name}. Average AUC: {avg_score}")

    logger.log("\nFinal Results:")
    for m, s in results.items():
        logger.log(f"  {m}: {s}")
