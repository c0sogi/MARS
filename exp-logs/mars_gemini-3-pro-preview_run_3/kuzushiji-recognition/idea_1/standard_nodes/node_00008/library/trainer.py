import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from library import config, dataset, model, loss, utils


class Trainer:
    def __init__(self, debug=False, load_cached_data=True):
        """
        Initializes the Trainer with model, data loaders, optimizer, and loss function.
        """
        self.debug = debug
        self.device = config.DEVICE

        # Ensure directories exist
        utils.setup_directories()

        # Set seeds for reproducibility
        utils.seed_everything(config.SEED)

        # Initialize Datasets and Loaders
        self.train_dataset = dataset.KuzushijiDataset(
            split="train", load_cached_data=load_cached_data, debug=debug
        )
        self.val_dataset = dataset.KuzushijiDataset(
            split="val", load_cached_data=load_cached_data, debug=debug
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # Initialize Model
        self.model = model.SparseCenterNet().to(self.device)

        # Initialize Loss (requires access to the classifier head)
        self.criterion = loss.SparseCenterNetLoss(self.model.classifier).to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)

        # Initialize Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
        )

        # Checkpoint path
        self.save_path = os.path.join(config.CACHE_DIR, "best_model.pth")
        self.best_val_loss = float("inf")

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_stats = {"loss": 0.0, "loss_hm": 0.0, "loss_reg": 0.0, "loss_cls": 0.0}
        num_batches = len(self.train_loader)

        for batch in self.train_loader:
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch["image"])

            # Compute loss
            total_loss, stats = self.criterion(outputs, batch)

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            # Accumulate stats
            for k, v in stats.items():
                running_stats[k] += v

        # Average stats
        avg_stats = {k: v / num_batches for k, v in running_stats.items()}
        return avg_stats

    def evaluate(self):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_stats = {"loss": 0.0, "loss_hm": 0.0, "loss_reg": 0.0, "loss_cls": 0.0}
        num_batches = len(self.val_loader)

        if num_batches == 0:
            return running_stats

        with torch.no_grad():
            for batch in self.val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                outputs = self.model(batch["image"])
                total_loss, stats = self.criterion(outputs, batch)

                for k, v in stats.items():
                    running_stats[k] += v

        avg_stats = {k: v / num_batches for k, v in running_stats.items()}
        return avg_stats

    def fit(self, num_epochs=config.NUM_EPOCHS, patience=5):
        """
        Main training loop with early stopping.
        """
        print(f"Starting training for {num_epochs} epochs on {self.device}...")

        # Re-initialize scheduler if num_epochs differs from config default
        if num_epochs != config.NUM_EPOCHS:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=1e-6
            )

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Train
            train_stats = self.train_one_epoch(epoch)

            # Validate
            val_stats = self.evaluate()

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start_time

            # Log Metrics (Full Precision)
            print(
                f"Epoch {epoch}/{num_epochs} | Time: {elapsed:.2f}s | LR: {current_lr:.2e}"
            )
            print(
                f"  Train Loss: {train_stats['loss']} (HM: {train_stats['loss_hm']}, Reg: {train_stats['loss_reg']}, Cls: {train_stats['loss_cls']})"
            )
            print(
                f"  Val Loss:   {val_stats['loss']} (HM: {val_stats['loss_hm']}, Reg: {val_stats['loss_reg']}, Cls: {val_stats['loss_cls']})"
            )

            # Early Stopping Check
            val_loss = val_stats["loss"]
            if val_loss < self.best_val_loss:
                print(
                    f"  Validation loss improved from {self.best_val_loss} to {val_loss}. Saving model..."
                )
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"  Validation loss did not improve. Patience: {patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val Loss: {self.best_val_loss}")

    def predict_and_submit(self):
        """
        Generates predictions for the test set and saves them to submission.csv.
        """
        print("Generating submission...")

        # Load best model weights
        if os.path.exists(self.save_path):
            self.model.load_state_dict(
                torch.load(self.save_path, map_location=self.device)
            )
            print(f"Loaded best model from {self.save_path}")
        else:
            print("Warning: Best model not found. Using current weights.")

        self.model.eval()

        # Setup Test Loader
        test_dataset = dataset.KuzushijiDataset(
            split="test", load_cached_data=True, debug=self.debug
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Create inverse class mapping
        idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

        results = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)

                # Forward Pass
                hm, reg, emb = self.model(images)

                # 1. Heatmap Post-processing
                hm = torch.sigmoid(hm)

                # Max-pooling NMS
                pad = 1
                hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=pad)
                keep = (hmax == hm).float()
                hm = hm * keep

                B, C, H, W = hm.shape
                hm_flat = hm.view(B, -1)

                # Select Top K detections
                K = min(config.MAX_DETECTIONS, H * W)
                topk_scores, topk_inds = torch.topk(hm_flat, K)

                # 2. Gather Features at Top K locations
                # Expand indices for gathering: (B, K, Dim)

                # Regression (Offsets)
                reg_flat = reg.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
                reg_gathered = torch.gather(
                    reg_flat, 1, topk_inds.unsqueeze(2).expand(-1, -1, 2)
                )

                # Embeddings
                emb_flat = (
                    emb.permute(0, 2, 3, 1)
                    .contiguous()
                    .view(B, -1, config.HEAD_CHANNELS)
                )
                emb_gathered = torch.gather(
                    emb_flat,
                    1,
                    topk_inds.unsqueeze(2).expand(-1, -1, config.HEAD_CHANNELS),
                )

                # 3. Classification
                cls_logits = self.model.classifier(emb_gathered)  # (B, K, NumClasses)
                # Get class with max score
                cls_scores, cls_ids = torch.max(cls_logits, dim=2)

                # 4. Coordinate Refinement
                topk_ys = (topk_inds // W).float()
                topk_xs = (topk_inds % W).float()

                # Add predicted offsets
                topk_xs = topk_xs + reg_gathered[:, :, 0]
                topk_ys = topk_ys + reg_gathered[:, :, 1]

                # 5. Format Results
                for b in range(B):
                    img_id = batch["image_id"][b]
                    c = batch["center"][b].cpu().numpy()
                    s = batch["scale"][b].item()

                    # Filter by confidence threshold
                    mask = topk_scores[b] > config.CONF_THRESHOLD

                    if not mask.any():
                        results.append(f"{img_id},")
                        continue

                    valid_xs = topk_xs[b][mask]
                    valid_ys = topk_ys[b][mask]
                    valid_cls_ids = cls_ids[b][mask]

                    # Transform coordinates back to original image space
                    coords = torch.stack([valid_xs, valid_ys], dim=1).cpu().numpy()
                    trans_coords = utils.transform_preds(coords, c, s, (W, H))

                    label_strs = []
                    for i in range(len(valid_xs)):
                        cls_idx = valid_cls_ids[i].item()
                        uni = idx_to_class[cls_idx]
                        x = int(trans_coords[i, 0])
                        y = int(trans_coords[i, 1])
                        label_strs.append(f"{uni} {x} {y}")

                    results.append(f"{img_id},{' '.join(label_strs)}")

        # Save to CSV
        with open(config.SUBMISSION_FILE_PATH, "w") as f:
            f.write("image_id,labels\n")
            f.write("\n".join(results))

        print(f"Submission saved to {config.SUBMISSION_FILE_PATH}")
