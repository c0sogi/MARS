import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import NuScenesDataset
from library.modules import IoUAwareCenterPoint
from library.loss import IoUAwareLoss
from library.utils import transform_points


class Trainer:
    def __init__(self):
        self.device = Config.DEVICE

        # Initialize Model
        self.model = IoUAwareCenterPoint().to(self.device)

        # Initialize Loss
        self.loss_fn = IoUAwareLoss()

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        self.scheduler = None
        self.best_val_loss = float("inf")
        self.patience = 3
        self.patience_counter = 0

    def train_epoch(self, dataloader):
        self.model.train()
        losses = []

        # Use tqdm for progress tracking, but keep it clean
        pbar = tqdm(dataloader, desc="Training", leave=False, disable=True)

        for batch in dataloader:
            # Move data to device
            points = [p.to(self.device) for p in batch["points"]]
            targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model({"points": points})

            # Loss calculation
            loss, loss_dict = self.loss_fn(preds, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 35)

            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            losses.append(loss_dict["total_loss"])

        return np.mean(losses)

    def validate(self, dataloader):
        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch in dataloader:
                points = [p.to(self.device) for p in batch["points"]]
                targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

                preds = self.model({"points": points})
                loss, loss_dict = self.loss_fn(preds, targets)
                losses.append(loss_dict["total_loss"])

        return np.mean(losses)

    def fit(self, train_loader, val_loader, epochs=Config.MAX_EPOCHS):
        print(f"Starting training for {epochs} epochs...")

        # Initialize OneCycleLR Scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LR,
            total_steps=epochs * len(train_loader),
            pct_start=0.3,
            div_factor=10,
            final_div_factor=100,
        )

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}"
            )

            # Checkpointing and Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                save_path = os.path.join(Config.CKPT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        # Load best model for future use
        best_path = os.path.join(Config.CKPT_DIR, "best_model.pth")
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            print("Loaded best model checkpoint.")

    def predict(self, test_loader, output_path):
        print("Generating predictions...")
        self.model.eval()

        # Ensure dataset cache is available for coordinate transforms
        dataset = test_loader.dataset
        if not hasattr(dataset, "lookup_table"):
            dataset._load_or_build_cache(True)

        # Create lookup index for fast retrieval
        lookup = dataset.lookup_table.set_index("token")

        results = []

        # Grid parameters for decoding
        voxel_x = Config.VOXEL_SIZE[0] * Config.DOWN_RATIO
        voxel_y = Config.VOXEL_SIZE[1] * Config.DOWN_RATIO

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Inference"):
                points = [p.to(self.device) for p in batch["points"]]
                tokens = batch["metadata"]["tokens"]

                preds = self.model({"points": points})

                # --- Decoding ---
                hm = preds["hm"]  # (B, C, H, W)
                hm = torch.sigmoid(hm)

                # NMS via MaxPool
                pad = 1
                hmax = F.max_pool2d(hm, (3, 3), stride=1, padding=pad)
                keep = (hmax == hm).float()
                hm = hm * keep

                # Top K selection
                B, C, H, W = hm.shape
                topk = Config.TOP_K

                hm = hm.view(B, -1)
                scores, inds = torch.topk(hm, topk)

                clses = inds // (H * W)
                inds = inds % (H * W)

                # Helper to gather features at indices
                def gather_feat(feat):
                    feat = feat.permute(0, 2, 3, 1).contiguous()
                    feat = feat.view(B, -1, feat.size(3))
                    dim = feat.size(2)
                    ind_g = inds.unsqueeze(2).expand(B, topk, dim)
                    return feat.gather(1, ind_g)

                reg = gather_feat(preds["reg"])
                wh = gather_feat(preds["wh"])
                rot = gather_feat(preds["rot"])
                z = gather_feat(preds["z"])
                iou = gather_feat(preds["iou"])

                # Process each sample in batch
                for b in range(B):
                    token = tokens[b]

                    # Retrieve coordinate transform
                    if token in lookup.index:
                        row = lookup.loc[token]
                        w2s = np.array(row["world_to_sensor"]).reshape(4, 4)
                        # We need Sensor -> World
                        s2w = np.linalg.inv(w2s)
                    else:
                        # Fallback (should not happen)
                        s2w = np.eye(4)

                    b_scores = scores[b]
                    b_clses = clses[b]
                    b_reg = reg[b]
                    b_wh = wh[b]
                    b_rot = rot[b]
                    b_z = z[b]
                    b_iou = iou[b]

                    # 1. Filter by raw confidence first for efficiency
                    mask = b_scores > Config.CONF_THRESHOLD
                    if mask.sum() == 0:
                        results.append({"Id": token, "PredictionString": ""})
                        continue

                    # 2. Rectify Score using IoU Prediction
                    # Score = score^(1-alpha) * iou^alpha
                    alpha = Config.IOU_RECTIFIER_ALPHA
                    rect_scores = torch.pow(b_scores, 1 - alpha) * torch.pow(
                        b_iou.squeeze(), alpha
                    )

                    # 3. Filter by rectified score
                    mask = rect_scores > Config.CONF_THRESHOLD
                    if mask.sum() == 0:
                        results.append({"Id": token, "PredictionString": ""})
                        continue

                    # Apply mask
                    f_scores = rect_scores[mask]
                    f_clses = b_clses[mask]
                    f_reg = b_reg[mask]
                    f_wh = b_wh[mask]
                    f_rot = b_rot[mask]
                    f_z = b_z[mask]
                    f_inds = inds[b][mask]

                    # 4. Recover Box Parameters in Sensor Frame
                    ys = (f_inds // W).float()
                    xs = (f_inds % W).float()

                    # Center (x, y, z)
                    cx = (xs + f_reg[:, 0]) * voxel_x + Config.POINT_CLOUD_RANGE[0]
                    cy = (ys + f_reg[:, 1]) * voxel_y + Config.POINT_CLOUD_RANGE[1]
                    cz = f_z[:, 0]

                    # Dimensions
                    cw = torch.exp(f_wh[:, 0])
                    cl = torch.exp(f_wh[:, 1])
                    ch = torch.exp(f_wh[:, 2])

                    # Yaw
                    cyaw = torch.atan2(f_rot[:, 0], f_rot[:, 1])

                    # 5. Transform to Global Frame
                    centers = torch.stack([cx, cy, cz], dim=1).cpu().numpy()

                    # Apply rigid transform
                    centers_global = transform_points(
                        centers, trans=s2w[:3, 3], rot_mat=s2w[:3, :3], inverse=False
                    )

                    # Adjust Yaw: Global Yaw = Sensor Yaw + Yaw(Sensor->Global)
                    yaw_s2w = np.arctan2(s2w[1, 0], s2w[0, 0])
                    cyaw_global = cyaw.cpu().numpy() + yaw_s2w

                    # 6. Format Prediction String
                    pred_strs = []
                    for k in range(len(f_scores)):
                        cls_name = Config.CLASS_NAMES[f_clses[k]]
                        s = f_scores[k].item()
                        x, y, z = centers_global[k]
                        w, l, h = cw[k].item(), cl[k].item(), ch[k].item()
                        y_ang = cyaw_global[k]

                        # Format: score x y z w l h yaw class_name
                        pred_strs.append(
                            f"{s:.4f} {x:.4f} {y:.4f} {z:.4f} {w:.4f} {l:.4f} {h:.4f} {y_ang:.4f} {cls_name}"
                        )

                    results.append(
                        {"Id": token, "PredictionString": " ".join(pred_strs)}
                    )

        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")


def run():
    Config.set_seed()

    # Initialize Datasets
    # Enable augmentation for training, disable for val/test
    # Enable targets for train/val, disable for test
    train_ds = NuScenesDataset("train", enable_augmentation=True, has_targets=True)
    val_ds = NuScenesDataset("val", enable_augmentation=False, has_targets=True)
    test_ds = NuScenesDataset("test", enable_augmentation=False, has_targets=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NuScenesDataset.collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NuScenesDataset.collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NuScenesDataset.collate_fn,
        pin_memory=True,
    )

    # Initialize and Run Trainer
    trainer = Trainer()
    trainer.fit(train_loader, val_loader, epochs=Config.MAX_EPOCHS)
    trainer.predict(test_loader, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    run()
