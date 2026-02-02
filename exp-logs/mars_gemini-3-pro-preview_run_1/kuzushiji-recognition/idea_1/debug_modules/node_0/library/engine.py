import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import KuzushijiDataset
from library.model import DKN
from library.loss import DKNLoss
from library.utils import (
    decode_outputs,
    get_affine_transform,
    load_unicode_map,
    seed_everything,
)


class Engine:
    def __init__(self):
        # Ensure reproducibility
        seed_everything(Config.SEED)

        self.device = torch.device(Config.DEVICE)
        self.model = DKN(num_classes=Config.NUM_CLASSES).to(self.device)
        self.criterion = DKNLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2, verbose=False
        )

        self.best_loss = float("inf")
        self.char_to_id, self.id_to_char = load_unicode_map()

        # Paths
        self.model_save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_one_epoch(self, dataloader, epoch, max_batches=None):
        self.model.train()
        running_loss = 0.0
        running_stats = {"loss_hm": 0.0, "loss_cls": 0.0, "loss_reg": 0.0}
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(batch["input"])
            loss, stats = self.criterion(outputs, batch)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            for k, v in stats.items():
                running_stats[k] += v

            num_batches += 1

        avg_loss = running_loss / num_batches if num_batches > 0 else 0
        avg_stats = (
            {k: v / num_batches for k, v in running_stats.items()}
            if num_batches > 0
            else running_stats
        )

        print(f"Epoch {epoch} Train Loss: {avg_loss}")
        print(f"Epoch {epoch} Train Stats: {avg_stats}")

        return avg_loss

    def evaluate(self, dataloader, epoch, max_batches=None):
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break

                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                outputs = self.model(batch["input"])
                loss, _ = self.criterion(outputs, batch)
                running_loss += loss.item()
                num_batches += 1

        avg_loss = running_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch} Val Loss: {avg_loss}")
        return avg_loss

    def fit(self, epochs=Config.NUM_EPOCHS, patience=5, debug=False):
        # Determine dataset parameters based on debug flag
        max_batches = 10 if debug else None

        # Load Datasets
        train_ds = KuzushijiDataset(split="train")
        val_ds = KuzushijiDataset(split="val")

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Starting training for {epochs} epochs (Debug={debug})...")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch, max_batches)
            val_loss = self.evaluate(val_loader, epoch, max_batches)

            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"New best model saved with val loss: {val_loss}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print("Training complete.")

    def predict(self):
        print("Starting inference generation...")

        if not os.path.exists(self.model_save_path):
            print(f"Error: Model file {self.model_save_path} not found.")
            return

        # Load best model
        self.model.load_state_dict(
            torch.load(self.model_save_path, map_location=self.device)
        )
        self.model.eval()

        test_ds = KuzushijiDataset(split="test")
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        results = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input"].to(self.device)

                # Forward Pass
                outputs = self.model(inputs)

                # Decode Outputs (Heatmap -> Coordinates & Classes)
                # Returns coords in output feature map space (256x256)
                xs, ys, scores, cls_ids = decode_outputs(
                    outputs["hm"], outputs["cls"], outputs["reg"], K=Config.TOP_K
                )

                batch_size = inputs.size(0)
                for i in range(batch_size):
                    image_id = batch["image_id"][i]
                    center = batch["center"][i].numpy()
                    scale = batch["scale"][i].item()

                    # Filter by confidence threshold
                    valid_mask = scores[i] > Config.CONF_THRESHOLD

                    valid_xs = xs[i][valid_mask].cpu().numpy()
                    valid_ys = ys[i][valid_mask].cpu().numpy()
                    valid_cls = cls_ids[i][valid_mask].cpu().numpy()

                    # 1. Map from Output Grid (256) to Input Image (1024)
                    # Feature map stride is 4
                    pts_input = np.stack([valid_xs, valid_ys], axis=1) * 4.0

                    # 2. Map from Input Image (1024) to Original Image
                    # Get Inverse Affine Transform
                    trans_inv = get_affine_transform(
                        center, scale, 0, [Config.IMG_SIZE, Config.IMG_SIZE], inv=True
                    )

                    label_strs = []
                    if len(pts_input) > 0:
                        # Vectorized Affine Transform
                        # Add homogeneous coordinate
                        pts_homo = np.concatenate(
                            [pts_input, np.ones((pts_input.shape[0], 1))], axis=1
                        )  # (N, 3)
                        # Apply matrix: (2, 3) x (3, N) -> (2, N)
                        pts_orig = (trans_inv @ pts_homo.T).T  # (N, 2)

                        for j in range(len(pts_orig)):
                            cid = valid_cls[j]
                            x_orig = int(pts_orig[j, 0])
                            y_orig = int(pts_orig[j, 1])

                            unicode_char = self.id_to_char.get(cid, None)
                            if unicode_char:
                                label_strs.append(f"{unicode_char} {x_orig} {y_orig}")

                    labels_joined = " ".join(label_strs)
                    results.append({"image_id": image_id, "labels": labels_joined})

        # Save Submission
        df_sub = pd.DataFrame(results)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
