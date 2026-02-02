import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, unpad_image, calc_map_score
from library.losses import BCEDiceLoss, LovaszHingeLoss, DeepSupervisionLoss
from library.dataset import SaltDataset
from library.model import SaltUNetPlusPlus


class Trainer:
    def __init__(self, debug=False):
        self.debug = debug
        self.device = torch.device(Config.DEVICE)

        # Reproducibility
        seed_everything(Config.SEED)

        # Initialize Model
        self.model = SaltUNetPlusPlus(
            encoder_name=Config.ENCODER_NAME,
            in_channels=Config.IN_CHANNELS,
            deep_supervision=Config.DEEP_SUPERVISION,
        )
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.MIN_LR,
        )

        # Losses
        self.bce_dice_loss = BCEDiceLoss()
        self.deep_supervision_loss = DeepSupervisionLoss(self.bce_dice_loss)
        self.lovasz_loss = LovaszHingeLoss(per_image=True)

        # DataLoaders
        self._init_dataloaders()

        # State
        self.best_map = 0.0
        self.start_epoch = 0

    def _init_dataloaders(self):
        train_dataset = SaltDataset(mode="train", debug=self.debug)
        val_dataset = SaltDataset(mode="val", debug=self.debug)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        # Determine Loss Function based on Schedule
        use_lovasz = epoch >= Config.LOVASZ_EPOCH_START
        loss_name = "Lovasz-Hinge" if use_lovasz else "BCE+Dice (DeepSup)"

        for i, (images, masks, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            outputs = self.model(images)

            # Loss Calculation
            if use_lovasz:
                # Fine-tuning: Apply Lovasz only to the final output
                # outputs is a list [out1, out2, out3, out4], take the last one
                final_output = outputs[-1]
                loss = self.lovasz_loss(final_output, masks)
            else:
                # Warm-up: Apply BCE+Dice to all deep supervision outputs
                loss = self.deep_supervision_loss(outputs, masks)

            # Backward
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader), loss_name

    def validate(self):
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, masks, _ in self.val_loader:
                images = images.to(self.device)

                # Forward (Eval mode returns only the final tensor)
                logits = self.model(images)
                probs = torch.sigmoid(logits)

                # Move to CPU
                probs = probs.cpu().numpy()
                masks = masks.numpy()

                # Unpad to original size (101x101) for accurate metric calculation
                for p, m in zip(probs, masks):
                    # p shape: (1, 128, 128), m shape: (1, 128, 128)
                    # unpad expects (H, W) or (H, W, C)
                    # Permute to (H, W, C) for unpad_image
                    p_t = np.transpose(p, (1, 2, 0))
                    m_t = np.transpose(m, (1, 2, 0))

                    p_orig = unpad_image(p_t, Config.ORIG_SIZE)
                    m_orig = unpad_image(m_t, Config.ORIG_SIZE)

                    # Squeeze back to (H, W)
                    all_preds.append(p_orig.squeeze())
                    all_targets.append(m_orig.squeeze())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Threshold Optimization
        # Sweep probability thresholds to find the one that maximizes mAP
        thresholds = np.arange(0.3, 0.75, 0.05)
        best_score = 0.0
        best_thresh = 0.5

        for t in thresholds:
            binary_preds = (all_preds > t).astype(np.uint8)
            score = calc_map_score(binary_preds, all_targets)
            if score > best_score:
                best_score = score
                best_thresh = t

        return best_score, best_thresh

    def start(self):
        print(f"Starting training on {self.device}...")
        print(
            f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}"
        )

        start_time = time.time()

        for epoch in range(self.start_epoch, Config.EPOCHS):
            epoch_start = time.time()

            # Train
            train_loss, loss_name = self.train_epoch(epoch)

            # Validate
            val_map, best_thresh = self.validate()

            # Scheduler Step
            self.scheduler.step(val_map)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Checkpoint
            saved_msg = ""
            if val_map > self.best_map:
                self.best_map = val_map
                torch.save(
                    self.model.state_dict(),
                    os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
                )
                saved_msg = "[Saved Best]"

            # Logging
            elapsed = time.time() - epoch_start
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Loss: {train_loss:.4f} ({loss_name}) | "
                f"Val mAP: {val_map:.10f} (Thresh: {best_thresh:.2f}) | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.0f}s {saved_msg}"
            )

            # Early Stopping check (optional, based on LR)
            if current_lr < Config.MIN_LR:
                print("Learning rate reached minimum. Stopping early.")
                break

        total_time = time.time() - start_time
        print(
            f"Training complete. Best mAP: {self.best_map:.10f}. Total time: {total_time/60:.1f}m"
        )
