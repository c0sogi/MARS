import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from library.config import Config
from library.utils import AverageMeter, calculate_roc_auc, save_checkpoint, set_seed
from library.model import get_model
from library.dataset import get_dataloaders


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device=Config.DEVICE,
        num_classes=Config.NUM_CLASSES,
        mixup_alpha=Config.MIXUP_ALPHA,
        swa_start_epoch=Config.SWA_START_EPOCH,
        swa_lr=Config.SWA_LR,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # SWA Components
        self.swa_start_epoch = swa_start_epoch
        self.swa_model = AveragedModel(model).to(device)
        self.swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

        # Tracking
        self.best_auc = 0.0

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()

        for i, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Mixup
            if self.mixup_alpha > 0:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                index = torch.randperm(images.size(0)).to(self.device)

                mixed_images = lam * images + (1 - lam) * images[index]
                mixed_labels = lam * labels + (1 - lam) * labels[index]

                outputs = self.model(mixed_images)
                loss = self.criterion(outputs, mixed_labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self, loader, model_to_validate=None):
        if model_to_validate is None:
            model_to_validate = self.model

        model_to_validate.eval()
        losses = AverageMeter()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model_to_validate(images)
                loss = self.criterion(outputs, labels)

                # Apply sigmoid for predictions
                preds = torch.sigmoid(outputs)

                losses.update(loss.item(), images.size(0))
                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        auc = calculate_roc_auc(all_targets, all_preds)
        return losses.avg, auc

    def fit(self, epochs, checkpoint_dir):
        print(f"Starting training for {epochs} epochs...")
        print(f"SWA will start at epoch {self.swa_start_epoch}")

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # SWA Logic
            is_swa_phase = epoch >= self.swa_start_epoch

            if is_swa_phase:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                lr = self.swa_scheduler.get_last_lr()[0]
            else:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]

            # Validate (Standard Model)
            val_loss, val_auc = self.validate(self.val_loader, self.model)

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"LR: {lr:.6f} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc:.10f} | "
                f"Time: {elapsed:.2f}s | "
                f"SWA: {'Active' if is_swa_phase else 'Inactive'}"
            )

            # Save Checkpoints
            # 1. Save Last
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_score": self.best_auc,
                },
                is_best=False,
                filepath=os.path.join(checkpoint_dir, "model_last.pth"),
            )

            # 2. Save Best (Only based on standard model performance during non-SWA or SWA)
            # Note: We continue tracking best standard model even during SWA phase
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "best_score": self.best_auc,
                    },
                    is_best=True,  # Creates model_best.pth
                    filepath=os.path.join(checkpoint_dir, "checkpoint.pth"),
                )

        # End of Training: Finalize SWA
        print("\nFinalizing SWA Model...")
        # Update BatchNorm statistics for SWA model
        update_bn(self.train_loader, self.swa_model, device=self.device)

        # Validate SWA Model
        swa_loss, swa_auc = self.validate(self.val_loader, self.swa_model)
        print(
            f"SWA Final Results -> Val Loss: {swa_loss:.6f} | Val AUC: {swa_auc:.10f}"
        )

        # Save SWA Model
        save_checkpoint(
            {
                "epoch": epochs,
                "state_dict": self.swa_model.state_dict(),
                "best_score": swa_auc,
            },
            is_best=False,
            filepath=os.path.join(checkpoint_dir, "model_swa.pth"),
        )

        return self.best_auc, swa_auc


def run_training(
    pseudo_labels_df=None,
    checkpoint_dir=Config.CHECKPOINT_DIR,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    seed=Config.SEED,
):
    """
    Main entry point to run training.

    Args:
        pseudo_labels_df (pd.DataFrame, optional): Pseudo-labels for student training.
        checkpoint_dir (str): Directory to save checkpoints.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size.
        seed (int): Random seed.

    Returns:
        tuple: (best_auc_standard, auc_swa)
    """
    set_seed(seed)

    # DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        train_metadata=Config.TRAIN_METADATA,
        val_metadata=Config.VAL_METADATA,
        test_metadata=Config.TEST_METADATA,
        pseudo_labels_df=pseudo_labels_df,
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
    )

    # Model
    model = get_model(
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        device=Config.DEVICE,
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler (Cosine Annealing for the main phase)
    # We set T_max to SWA_START_EPOCH because we switch schedulers afterwards
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.SWA_START_EPOCH,
        eta_min=Config.SWA_LR,  # Decay down to SWA LR
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        num_classes=Config.NUM_CLASSES,
        mixup_alpha=Config.MIXUP_ALPHA,
        swa_start_epoch=Config.SWA_START_EPOCH,
        swa_lr=Config.SWA_LR,
    )

    # Execute Training
    best_auc, swa_auc = trainer.fit(epochs=epochs, checkpoint_dir=checkpoint_dir)

    return best_auc, swa_auc
