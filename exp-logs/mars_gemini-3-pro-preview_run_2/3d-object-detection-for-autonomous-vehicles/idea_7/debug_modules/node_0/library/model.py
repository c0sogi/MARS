import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import math

from library.config import Config
from library.dataset import NuScenesDataset
from library.modules import IoUAwareCenterPoint
from library.utils import iou3d_global, transform_points

# ==============================================================================
# Loss Functions
# ==============================================================================


def _neg_loss(pred, gt):
    """Modified focal loss. Exact same as CornerNet.
    Runs on a batch of heatmaps.
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class CenterPointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = _neg_loss
        self.crit_reg = torch.nn.L1Loss(reduction="none")

    def forward(self, preds, targets):
        # Heatmap Loss
        hm_loss = self.crit(preds["hm"], targets["hm"])

        target_mask = targets["mask"]
        ind = targets["ind"]

        # Helper to calculate reg loss
        def get_reg_loss(name):
            if name not in preds:
                return torch.tensor(0.0, device=preds["hm"].device)
            pred = _transpose_and_gather_feat(preds[name], ind)
            target = targets[name]
            mask = target_mask.unsqueeze(2).expand_as(target).float()
            loss = self.crit_reg(pred, target) * mask
            loss = loss.sum() / (mask.sum() + 1e-4)
            return loss

        reg_loss = get_reg_loss("reg")
        wh_loss = get_reg_loss("wh")
        rot_loss = get_reg_loss("rot")
        z_loss = get_reg_loss("z")

        # IoU Loss
        # We calculate the actual IoU between the predicted box (detached) and the GT box
        # and use this as the target for the 'iou' head.

        # 1. Reconstruct boxes from preds (detached) and targets (GT)
        pred_reg = _transpose_and_gather_feat(preds["reg"], ind).detach()
        pred_wh = _transpose_and_gather_feat(preds["wh"], ind).detach()
        pred_rot = _transpose_and_gather_feat(preds["rot"], ind).detach()
        pred_z = _transpose_and_gather_feat(preds["z"], ind).detach()

        gt_reg = targets["reg"]
        gt_wh = targets["wh"]
        gt_rot = targets["rot"]
        gt_z = targets["z"]

        # Grid parameters
        B, K = ind.shape
        W = preds["hm"].shape[3]

        xs = (ind % W).float()
        ys = (ind // W).float()

        voxel_x = Config.VOXEL_SIZE[0] * Config.DOWN_RATIO
        voxel_y = Config.VOXEL_SIZE[1] * Config.DOWN_RATIO

        # Construct boxes for IoU calculation (x, y, z, w, l, h, yaw)
        def get_boxes(reg, wh, z_v, rot):
            bx = (xs + reg[..., 0]) * voxel_x + Config.POINT_CLOUD_RANGE[0]
            by = (ys + reg[..., 1]) * voxel_y + Config.POINT_CLOUD_RANGE[1]
            bw = torch.exp(wh[..., 0])
            bl = torch.exp(wh[..., 1])
            bh = torch.exp(wh[..., 2])
            bz = z_v[..., 0]
            brot = torch.atan2(rot[..., 0], rot[..., 1])
            return torch.stack([bx, by, bz, bw, bl, bh, brot], dim=-1)

        p_boxes_all = get_boxes(pred_reg, pred_wh, pred_z, pred_rot)
        g_boxes_all = get_boxes(gt_reg, gt_wh, gt_z, gt_rot)

        # Filter valid objects
        mask = target_mask.bool().view(-1)
        if mask.sum() > 0:
            p_boxes = p_boxes_all.view(-1, 7)[mask]
            g_boxes = g_boxes_all.view(-1, 7)[mask]

            # Calculate IoU (N, N) -> take diagonal
            # iou3d_global returns tensor on same device
            ious = iou3d_global(p_boxes, g_boxes)
            iou_targets = torch.diag(ious)

            # Predict IoU
            pred_iou = _transpose_and_gather_feat(preds["iou"], ind).view(-1, 1)
            pred_iou = pred_iou[mask.view(B, K)].squeeze()

            iou_loss = F.l1_loss(pred_iou, iou_targets)
        else:
            iou_loss = torch.tensor(0.0, device=preds["hm"].device)

        total_loss = (
            Config.LOSS_WEIGHTS["hm"] * hm_loss
            + Config.LOSS_WEIGHTS["reg"] * reg_loss
            + Config.LOSS_WEIGHTS["wh"] * wh_loss
            + Config.LOSS_WEIGHTS["rot"] * rot_loss
            + Config.LOSS_WEIGHTS["z"] * z_loss
            + Config.LOSS_WEIGHTS["iou"] * iou_loss
        )

        return total_loss, {
            "hm_loss": hm_loss.item(),
            "reg_loss": reg_loss.item(),
            "wh_loss": wh_loss.item(),
            "rot_loss": rot_loss.item(),
            "z_loss": z_loss.item(),
            "iou_loss": iou_loss.item(),
            "total_loss": total_loss.item(),
        }


# ==============================================================================
# Model Runner
# ==============================================================================


class ObjectDetectionModel:
    def __init__(self):
        self.device = Config.DEVICE
        self.model = IoUAwareCenterPoint().to(self.device)
        self.loss_fn = CenterPointLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )
        self.scheduler = None  # Initialized in fit()

        self.best_val_loss = float("inf")
        self.patience = 3
        self.patience_counter = 0

    def train_epoch(self, dataloader):
        self.model.train()
        losses = []

        for batch in tqdm(dataloader, desc="Training", leave=False):
            points = [p.to(self.device) for p in batch["points"]]
            targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

            self.optimizer.zero_grad()
            preds = self.model({"points": points})

            loss, loss_dict = self.loss_fn(preds, targets)

            loss.backward()
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
            for batch in tqdm(dataloader, desc="Validation", leave=False):
                points = [p.to(self.device) for p in batch["points"]]
                targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

                preds = self.model({"points": points})
                loss, loss_dict = self.loss_fn(preds, targets)
                losses.append(loss_dict["total_loss"])

        return np.mean(losses)

    def fit(self, train_loader, val_loader, epochs):
        print(f"Starting training for {epochs} epochs...")

        # OneCycleLR
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
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    os.path.join(Config.CKPT_DIR, "best_model.pth"),
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        # Load best model
        best_path = os.path.join(Config.CKPT_DIR, "best_model.pth")
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path))
            print("Loaded best model for inference.")

    def predict(self, test_loader, output_path):
        print("Generating predictions...")
        self.model.eval()

        # Access lookup table for coordinate transforms
        dataset = test_loader.dataset
        # Ensure cache is loaded
        if not hasattr(dataset, "lookup_table"):
            dataset._load_or_build_cache(True)

        lookup = dataset.lookup_table.set_index("token")

        results = []

        voxel_x = Config.VOXEL_SIZE[0] * Config.DOWN_RATIO
        voxel_y = Config.VOXEL_SIZE[1] * Config.DOWN_RATIO

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Inference"):
                points = [p.to(self.device) for p in batch["points"]]
                tokens = batch["metadata"]["tokens"]

                preds = self.model({"points": points})

                # Decode Heatmap
                hm = preds["hm"]  # (B, C, H, W)
                hm = torch.sigmoid(hm)

                # Simple NMS using MaxPool
                pad = 1
                hmax = F.max_pool2d(hm, (3, 3), stride=1, padding=pad)
                keep = (hmax == hm).float()
                hm = hm * keep

                # Top K
                B, C, H, W = hm.shape
                topk = Config.TOP_K

                # Flatten
                hm = hm.view(B, -1)
                scores, inds = torch.topk(hm, topk)

                clses = inds // (H * W)
                inds = inds % (H * W)

                # Gather regressions
                # Helper to gather
                def gather(feat):
                    feat = (
                        feat.permute(0, 2, 3, 1).contiguous().view(B, -1, feat.size(3))
                    )
                    dim = feat.size(2)
                    ind_g = inds.unsqueeze(2).expand(B, topk, dim)
                    return feat.gather(1, ind_g)

                reg = gather(preds["reg"])
                wh = gather(preds["wh"])
                rot = gather(preds["rot"])
                z = gather(preds["z"])
                iou = gather(preds["iou"])

                # Process batch
                for b in range(B):
                    token = tokens[b]

                    # Get Transform
                    if token in lookup.index:
                        row = lookup.loc[token]
                        w2s = np.array(row["world_to_sensor"]).reshape(4, 4)
                        s2w = np.linalg.inv(w2s)
                    else:
                        s2w = np.eye(4)  # Should not happen

                    # Parse predictions
                    b_scores = scores[b]
                    b_clses = clses[b]
                    b_reg = reg[b]
                    b_wh = wh[b]
                    b_rot = rot[b]
                    b_z = z[b]
                    b_iou = iou[b]

                    # Filter by confidence
                    mask = b_scores > Config.CONF_THRESHOLD
                    if mask.sum() == 0:
                        results.append({"Id": token, "PredictionString": ""})
                        continue

                    # Rectify Score
                    # Score = score^(1-alpha) * iou^alpha
                    alpha = Config.IOU_RECTIFIER_ALPHA
                    rect_scores = torch.pow(b_scores, 1 - alpha) * torch.pow(
                        b_iou.squeeze(), alpha
                    )

                    # Filter
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

                    # Recover Box Parameters in Sensor Frame
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

                    # Transform to Global
                    # Points: (N, 3)
                    centers = torch.stack([cx, cy, cz], dim=1).cpu().numpy()

                    # Apply rotation and translation
                    # P_global = P_sensor @ R_s2w.T + T_s2w
                    # transform_points handles this
                    centers_global = transform_points(
                        centers, s2w[:3, 3], rot_mat=s2w[:3, :3], inverse=False
                    )

                    # Yaw Transform
                    # Global Yaw = Sensor Yaw + Yaw(Sensor->Global)
                    # Extract yaw from s2w rotation matrix
                    # yaw_s2w = atan2(R[1,0], R[0,0])
                    yaw_s2w = np.arctan2(s2w[1, 0], s2w[0, 0])
                    cyaw_global = cyaw.cpu().numpy() + yaw_s2w

                    # Format String
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


# ==============================================================================
# Main Execution Pipeline
# ==============================================================================


def run_pipeline():
    Config.set_seed()

    # 1. Datasets
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

    # 2. Model
    runner = ObjectDetectionModel()

    # 3. Train
    runner.fit(train_loader, val_loader, epochs=Config.MAX_EPOCHS)

    # 4. Predict
    runner.predict(test_loader, Config.SUBMISSION_PATH)


# Execute
run_pipeline()
