import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import numpy as np

from library.config import Config
from library.dataset import WhaleDataset, get_transforms
from library.model import WhaleDenseNet
from library.loss import WhaleLoss
from library.utils import AverageMeter, calculate_map5, save_checkpoint, seed_everything


class Trainer:
    """
    Manages the training lifecycle including:
    - Data loading
    - Model initialization (DenseNet121 + ArcFace)
    - Optimization (AdamW + SWA)
    - Training loop with SWA logic
    - Validation with Test-Time Augmentation (TTA)
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.device = Config.DEVICE

        # ---------------------------------------------------------------------
        # Data Loading
        # ---------------------------------------------------------------------
        print("Initializing Datasets...")
        self.train_dataset = WhaleDataset(
            mode="train", transform=get_transforms("train"), load_cached_data=True
        )
        self.val_dataset = WhaleDataset(
            mode="val", transform=get_transforms("val"), load_cached_data=True
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # ---------------------------------------------------------------------
        # Model & Optimization
        # ---------------------------------------------------------------------
        print("Initializing Model...")
        self.model = WhaleDenseNet(
            backbone_name=Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            embedding_size=Config.EMBEDDING_SIZE,
            num_classes=Config.NUM_CLASSES,
            s=Config.ARCFACE_SCALE,
            m=Config.ARCFACE_MARGIN,
        ).to(self.device)

        self.criterion = WhaleLoss(label_smoothing=Config.LABEL_SMOOTHING).to(
            self.device
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # ---------------------------------------------------------------------
        # Schedulers & SWA
        # ---------------------------------------------------------------------
        # Phase 1 Scheduler: Cosine Annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.SWA_START_EPOCH, eta_min=Config.MIN_LR
        )

        # SWA Model
        self.swa_model = AveragedModel(self.model).to(self.device)

        # Phase 2 Scheduler: SWA LR (Cyclic)
        self.swa_scheduler = SWALR(
            self.optimizer, swa_lr=Config.SWA_LR, anneal_epochs=1, anneal_strategy="cos"
        )

        self.start_epoch = 1
        self.best_map5 = 0.0

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()

        # Determine which scheduler to use for logging current LR
        current_lr = self.optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch}/{Config.TOTAL_EPOCHS} [Train] LR: {current_lr:.6f}")

        for i, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            # Note: ArcFace requires labels during training
            logits = self.model(images, labels)
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        print(f"Epoch {epoch} Train Loss: {losses.avg:.6f}")

    def validate(self, epoch, model_to_validate):
        """
        Validates the model using Test-Time Augmentation (TTA).
        TTA: Average predictions of original and horizontally flipped images.
        """
        model_to_validate.eval()

        all_preds = []
        all_targets = []

        print(f"Epoch {epoch} [Val] Running validation with TTA...")

        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # TTA: 1. Original
                logits_orig = model_to_validate(
                    images, labels=None
                )  # labels=None for inference mode

                # TTA: 2. Horizontal Flip
                if Config.TTA_FLIP:
                    images_flipped = torch.flip(images, dims=[3])
                    logits_flip = model_to_validate(images_flipped, labels=None)
                    logits = (logits_orig + logits_flip) / 2.0
                else:
                    logits = logits_orig

                # Get Top K predictions
                # logits shape: (B, num_classes)
                # We need indices of top 5
                _, top_indices = torch.topk(logits, k=Config.TOP_K, dim=1)

                all_preds.append(top_indices.cpu())
                all_targets.append(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        map5 = calculate_map5(all_preds, all_targets)
        print(f"Epoch {epoch} Val MAP@5: {map5}")

        return map5

    def fit(self):
        print(f"Starting training for {Config.TOTAL_EPOCHS} epochs.")
        print(f"SWA starts at epoch {Config.SWA_START_EPOCH + 1}")

        for epoch in range(self.start_epoch, Config.TOTAL_EPOCHS + 1):
            start_time = time.time()

            # 1. Train
            self.train_one_epoch(epoch)

            # 2. Scheduler Step & SWA Update
            if epoch <= Config.SWA_START_EPOCH:
                # Phase 1: Standard Cosine Annealing
                self.scheduler.step()
                is_swa_phase = False
            else:
                # Phase 2: SWA
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                is_swa_phase = True
                print(f"Epoch {epoch}: SWA parameters updated.")

            # 3. Validation
            # We validate the current base model to monitor progress.
            # SWA model is usually validated at the end after BN update.
            val_map5 = self.validate(epoch, self.model)

            # 4. Save Checkpoint (Base Model)
            is_best = val_map5 > self.best_map5
            if is_best:
                self.best_map5 = val_map5

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "best_map5": self.best_map5,
                    "optimizer": self.optimizer.state_dict(),
                },
                is_best=is_best,
                filename=f"checkpoint_epoch_{epoch}.pth.tar",
            )

            elapsed = time.time() - start_time
            print(f"Epoch {epoch} completed in {elapsed:.1f}s")
            print("-" * 40)

        # ---------------------------------------------------------------------
        # Finalize SWA Model
        # ---------------------------------------------------------------------
        print("\nTraining complete. Finalizing SWA model...")

        # Update BatchNorm statistics for the SWA model using training data
        print("Updating SWA BatchNorm statistics...")
        update_bn(self.train_loader, self.swa_model, device=self.device)

        # Validate SWA Model
        print("Validating SWA model...")
        swa_map5 = self.validate("SWA_Final", self.swa_model)

        # Save SWA Model
        print(f"Saving SWA model (MAP@5: {swa_map5})...")
        save_checkpoint(
            {
                "epoch": Config.TOTAL_EPOCHS,
                "state_dict": self.swa_model.state_dict(),  # Note: AveragedModel wraps the module
                "map5": swa_map5,
            },
            is_best=False,  # We save this specifically as the SWA model
            filename="swa_model_final.pth.tar",
        )

        # Also save as model_best if it outperforms the best base model
        if swa_map5 > self.best_map5:
            print(
                "SWA model outperformed best base model. Saving as model_best.pth.tar"
            )
            save_checkpoint(
                {
                    "epoch": Config.TOTAL_EPOCHS,
                    "state_dict": self.swa_model.state_dict(),
                    "best_map5": swa_map5,
                },
                is_best=True,
            )

        print("Done.")
