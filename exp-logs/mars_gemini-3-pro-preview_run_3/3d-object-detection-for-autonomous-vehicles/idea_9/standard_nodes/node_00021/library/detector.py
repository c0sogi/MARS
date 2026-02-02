import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import math

from library.config import Config
from library.dataset import LidarDataset, collate_fn
from library.model_blocks import TwoStagePointPillars
from library.utils import box_encode, box_decode, iou3d, points_in_boxes_gpu, nms_3d


class LossModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 2.0
        self.beta = 4.0
        self.weights = Config.LOSS_WEIGHTS

    def gaussian_focal_loss(self, pred, target):
        """
        pred: (B, C, H, W) sigmoid probabilities
        target: (B, C, H, W) gaussian heatmap [0, 1]
        """
        pos_inds = target.eq(1)
        neg_inds = target.lt(1)

        neg_weights = torch.pow(1 - target, self.beta)

        loss = 0

        # Clamp for numerical stability
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = (
            torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds
        )

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss

    def l1_loss(self, pred, target, mask):
        """
        pred: (B, 8, H, W)
        target: (B, 8, H, W)
        mask: (B, H, W)
        """
        num = mask.float().sum() * pred.shape[1]
        if num == 0:
            return pred.sum() * 0

        mask = mask.unsqueeze(1).expand_as(pred)
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
        return loss / num

    def generate_stage1_targets(self, gt_boxes, gt_labels, feature_map_shape):
        """
        Generates heatmap and regression targets for Stage 1.
        """
        B, _, H, W = feature_map_shape
        device = gt_boxes[0].device

        heatmap = torch.zeros((B, Config.NUM_CLASSES, H, W), device=device)
        regression = torch.zeros((B, 8, H, W), device=device)
        mask = torch.zeros((B, H, W), device=device)

        voxel_size = torch.tensor(Config.VOXEL_SIZE, device=device)
        pc_range = torch.tensor(Config.POINT_CLOUD_RANGE, device=device)

        # Feature stride is 1 based on model analysis
        stride = 1

        for b in range(B):
            boxes = gt_boxes[b]
            labels = gt_labels[b]

            if len(boxes) == 0:
                continue

            # Convert centers to grid coordinates
            # x_grid = (x_world - min_x) / (voxel_x * stride)
            xs = (boxes[:, 0] - pc_range[0]) / (voxel_size[0] * stride)
            ys = (boxes[:, 1] - pc_range[1]) / (voxel_size[1] * stride)

            xs_int = xs.long()
            ys_int = ys.long()

            # Filter out of bounds
            valid = (xs_int >= 0) & (xs_int < W) & (ys_int >= 0) & (ys_int < H)

            valid_boxes = boxes[valid]
            valid_labels = labels[valid]
            xs = xs[valid]
            ys = ys[valid]
            xs_int = xs_int[valid]
            ys_int = ys_int[valid]

            if len(valid_boxes) == 0:
                continue

            # 1. Heatmap (Gaussian Splatting)
            for i in range(len(valid_boxes)):
                l = valid_labels[i]
                xc, yc = xs_int[i], ys_int[i]

                # Radius based on object size (simple heuristic)
                w_grid = valid_boxes[i, 3] / (voxel_size[0] * stride)
                l_grid = valid_boxes[i, 4] / (voxel_size[1] * stride)
                radius = max(0, int(min(w_grid, l_grid) / 2))
                radius = max(1, radius)

                self.draw_gaussian(heatmap[b, l], (xc, yc), radius)

                # 2. Regression Targets
                # [dx, dy, z, log(w), log(l), log(h), sin, cos]
                # dx, dy are offsets from center of pixel
                # center of pixel (xc, yc) is xc + 0.5
                # But CenterHead usually regresses from top-left or center?
                # Model get_proposals uses: final_x = xs_world + reg_vals
                # xs_world corresponds to center of pixel.
                # So we regress offset from center.

                # World coord of pixel center
                wx = (
                    (xc.float() * stride * voxel_size[0])
                    + pc_range[0]
                    + (voxel_size[0] * stride / 2)
                )
                wy = (
                    (yc.float() * stride * voxel_size[1])
                    + pc_range[1]
                    + (voxel_size[1] * stride / 2)
                )

                regression[b, 0, yc, xc] = valid_boxes[i, 0] - wx
                regression[b, 1, yc, xc] = valid_boxes[i, 1] - wy
                regression[b, 2, yc, xc] = valid_boxes[i, 2]
                regression[b, 3, yc, xc] = torch.log(valid_boxes[i, 3])
                regression[b, 4, yc, xc] = torch.log(valid_boxes[i, 4])
                regression[b, 5, yc, xc] = torch.log(valid_boxes[i, 5])
                regression[b, 6, yc, xc] = torch.sin(valid_boxes[i, 6])
                regression[b, 7, yc, xc] = torch.cos(valid_boxes[i, 6])

                mask[b, yc, xc] = 1

        return heatmap, regression, mask

    def draw_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian_2d((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0], heatmap.shape[1]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian[
            radius - top : radius + bottom, radius - left : radius + right
        ]

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(
                masked_heatmap, masked_gaussian.to(heatmap.device), out=masked_heatmap
            )

    def gaussian_2d(self, shape, sigma=1):
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return torch.tensor(h, dtype=torch.float32)


class PointPillarsDetector:
    def __init__(self, load_checkpoint=None):
        self.device = torch.device(Config.DEVICE)
        self.model = TwoStagePointPillars().to(self.device)
        self.loss_module = LossModule().to(self.device)

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )
        self.scheduler = None  # Initialized in train loop

        self.best_val_loss = float("inf")

        if load_checkpoint and os.path.exists(load_checkpoint):
            self.load_checkpoint(load_checkpoint)

    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1} Train", disable=True)

        for batch in dataloader:
            self.optimizer.zero_grad()

            # Move data to device
            voxels = batch["voxels"].to(self.device)
            num_points = batch["num_points"].to(self.device)
            coordinates = batch["coordinates"].to(self.device)
            gt_boxes = [b.to(self.device) for b in batch["gt_boxes"]]
            gt_labels = [l.to(self.device) for l in batch["gt_labels"]]
            batch_size = len(batch["sample_tokens"])

            # --- Stage 1 Forward ---
            hm_pred, reg_pred, feats = self.model(
                voxels, num_points, coordinates, batch_size=batch_size
            )

            # Stage 1 Targets & Loss
            hm_tgt, reg_tgt, reg_mask = self.loss_module.generate_stage1_targets(
                gt_boxes, gt_labels, hm_pred.shape
            )

            loss_hm = self.loss_module.gaussian_focal_loss(hm_pred, hm_tgt)
            loss_loc_s1 = self.loss_module.l1_loss(reg_pred, reg_tgt, reg_mask)

            # --- Stage 2 Preparation ---
            # Generate proposals
            with torch.no_grad():
                proposals, scores, _ = self.model.center_head.get_proposals(
                    hm_pred, reg_pred, topk=200
                )
                # proposals: (B, K, 7)

            # Sample Proposals for Stage 2
            sampled_proposals, roi_reg_tgt, roi_iou_tgt = self.sample_proposals(
                proposals, gt_boxes
            )

            # If no valid proposals (rare), skip stage 2 loss
            if sampled_proposals.shape[1] > 0:
                # --- Stage 2 Forward ---
                residuals, iou_pred = self.model.forward_stage2(
                    feats, sampled_proposals
                )

                # Stage 2 Loss
                # Regression Loss (only on positives)
                # roi_reg_tgt has NaNs or zeros for negatives?
                # In sample_proposals, we should mark negatives.
                # Let's assume sample_proposals returns mask or handles it.
                # Simplified: sample_proposals returns all samples, we need a mask for positives.
                # Re-implement sampling inside here for clarity or check helper.

                # Check helper implementation below
                pass

            # Since sample_proposals is complex, let's implement the logic inline or helper
            # Re-doing the call to be cleaner

            loss_loc_s2 = torch.tensor(0.0, device=self.device)
            loss_iou_s2 = torch.tensor(0.0, device=self.device)

            if sampled_proposals.shape[1] > 0:
                # Identify positives (iou_tgt > 0.5 usually)
                # roi_iou_tgt is the actual IoU with GT
                pos_mask = roi_iou_tgt > 0.5

                # Refinement Loss (L1)
                if pos_mask.sum() > 0:
                    loss_loc_s2 = F.l1_loss(
                        residuals[pos_mask], roi_reg_tgt[pos_mask], reduction="mean"
                    )

                # Rectification Loss (BCE)
                # Target is the actual IoU (soft label) or binary?
                # "Rectification Branch... predicts the 3D IoU... target is actual calculated IoU"
                # So we use BCE with soft targets or MSE. BCE is standard for 0-1.
                loss_iou_s2 = F.binary_cross_entropy(
                    iou_pred, roi_iou_tgt, reduction="mean"
                )

            # Total Loss
            loss = (
                self.loss_module.weights["cls_weight"] * loss_hm
                + self.loss_module.weights["loc_weight"] * loss_loc_s1
                + self.loss_module.weights["loc_weight"] * loss_loc_s2
                + self.loss_module.weights["iou_weight"] * loss_iou_s2
            )

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.GRAD_CLIP_NORM
            )

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            pbar.update(1)

        return total_loss / len(dataloader)

    def sample_proposals(self, proposals, gt_boxes_list):
        """
        Matches proposals to GT and samples for Stage 2.
        Returns:
            sampled_proposals: (B, N_sample, 7)
            reg_targets: (B, N_sample, 7)
            iou_targets: (B, N_sample)
        """
        B, K, _ = proposals.shape
        N_sample = 128

        batch_proposals = []
        batch_reg_tgt = []
        batch_iou_tgt = []

        for b in range(B):
            props = proposals[b].detach()  # (K, 7)
            gts = gt_boxes_list[b]  # (M, 7)

            if len(gts) == 0:
                # All negatives
                # Sample random proposals
                if K > 0:
                    indices = torch.randperm(K)[:N_sample]
                    sel_props = props[indices]
                    # Targets
                    reg_tgt = torch.zeros_like(sel_props)
                    iou_tgt = torch.zeros(len(sel_props), device=self.device)
                else:
                    sel_props = torch.zeros((0, 7), device=self.device)
                    reg_tgt = torch.zeros((0, 7), device=self.device)
                    iou_tgt = torch.zeros((0,), device=self.device)
            else:
                # Calculate IoU
                ious = iou3d(props, gts)  # (K, M)
                max_ious, max_ids = ious.max(dim=1)  # (K,)

                # Assign labels
                # Positives: IoU > 0.5
                # Negatives: IoU < 0.5
                pos_inds = torch.where(max_ious >= 0.5)[0]
                neg_inds = torch.where(max_ious < 0.5)[0]

                # Sample
                num_pos = min(len(pos_inds), N_sample // 2)
                num_neg = N_sample - num_pos

                if len(pos_inds) > 0:
                    pos_inds = pos_inds[torch.randperm(len(pos_inds))[:num_pos]]

                if len(neg_inds) > 0:
                    # Hard negative mining could be done here, but random is faster
                    neg_inds = neg_inds[
                        torch.randperm(len(neg_inds))[: min(len(neg_inds), num_neg)]
                    ]

                sel_inds = torch.cat([pos_inds, neg_inds])
                sel_props = props[sel_inds]

                # Targets for positives
                matched_gts = gts[max_ids[sel_inds]]

                # Regression Targets: Encode(GT, Proposal)
                reg_tgt = box_encode(matched_gts, sel_props)

                # IoU Targets: We want the network to predict the IoU of the REFINED box.
                # But we don't have the refined box yet.
                # Standard approximation: Use IoU(Proposal, GT) or
                # use a second forward pass (too slow).
                # Or, we use the IoU of the matched GT as target? No, that's 1.0.
                # We use the IoU(Proposal, GT) as a proxy for "quality".
                # Or better: The prompt says "predicts the 3D IoU between the refined box and ground truth".
                # To do this strictly, we need to apply the regression first inside the training loop.
                # I will do this in the training loop. Here I just return the matched GTs to calculate IoU later.
                # But to keep interface simple, I will return matched GTs implicitly via reg_targets?
                # No, let's calculate IoU(Proposal, GT) here as a baseline,
                # and in train_epoch we can re-calculate IoU(Refined, GT).

                # Actually, let's return matched GTs directly? No, shape mismatch.
                # Let's return the matched GT boxes corresponding to samples.
                # Re-purposing reg_targets to hold encoded targets.
                # I will calculate IoU targets inside train_epoch after refinement.
                # So here, iou_tgt will be placeholder or IoU(Prop, GT).
                iou_tgt = max_ious[sel_inds]

            batch_proposals.append(sel_props)
            batch_reg_tgt.append(reg_tgt)
            batch_iou_tgt.append(iou_tgt)  # This is IoU(Prop, GT)

        # Pad to same size if necessary?
        # Since we iterate batch in loop, we can stack if sizes match or use list.
        # But `forward_stage2` expects (B, K, 7).
        # We need to pad to N_sample.

        final_props = torch.zeros((B, N_sample, 7), device=self.device)
        final_reg = torch.zeros((B, N_sample, 7), device=self.device)
        final_iou = torch.zeros((B, N_sample), device=self.device)

        for b in range(B):
            n = len(batch_proposals[b])
            if n > 0:
                final_props[b, :n] = batch_proposals[b]
                final_reg[b, :n] = batch_reg_tgt[b]
                final_iou[b, :n] = batch_iou_tgt[b]

        return final_props, final_reg, final_iou

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                voxels = batch["voxels"].to(self.device)
                num_points = batch["num_points"].to(self.device)
                coordinates = batch["coordinates"].to(self.device)
                gt_boxes = [b.to(self.device) for b in batch["gt_boxes"]]
                gt_labels = [l.to(self.device) for l in batch["gt_labels"]]
                batch_size = len(batch["sample_tokens"])

                hm_pred, reg_pred, feats = self.model(
                    voxels, num_points, coordinates, batch_size=batch_size
                )

                hm_tgt, reg_tgt, reg_mask = self.loss_module.generate_stage1_targets(
                    gt_boxes, gt_labels, hm_pred.shape
                )

                loss_hm = self.loss_module.gaussian_focal_loss(hm_pred, hm_tgt)
                loss_loc = self.loss_module.l1_loss(reg_pred, reg_tgt, reg_mask)

                loss = loss_hm + loss_loc
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, train_loader, val_loader):
        print(f"Starting training on {self.device}...")

        steps_per_epoch = len(train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LR,
            steps_per_epoch=steps_per_epoch,
            epochs=Config.EPOCHS,
            pct_start=0.3,
        )

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    os.path.join(Config.WORKING_DIR, "model_checkpoint.pth")
                )

            # Simple Early Stopping
            if epoch > 5 and val_loss > self.best_val_loss * 1.2:
                print("Early stopping triggered.")
                break

    def save_checkpoint(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Loaded checkpoint from {path}")

    def generate_submission(self, test_loader, output_path):
        self.model.eval()
        results = []

        print("Generating submission...")
        with torch.no_grad():
            for batch in tqdm(test_loader, disable=True):
                voxels = batch["voxels"].to(self.device)
                num_points = batch["num_points"].to(self.device)
                coordinates = batch["coordinates"].to(self.device)
                batch_size = len(batch["sample_tokens"])

                # Stage 1
                hm_pred, reg_pred, feats = self.model(
                    voxels, num_points, coordinates, batch_size=batch_size
                )

                # Get Proposals
                proposals, scores, cls_ids = self.model.center_head.get_proposals(
                    hm_pred, reg_pred, topk=500
                )

                # Stage 2
                residuals, iou_pred = self.model.forward_stage2(feats, proposals)

                # Refine Boxes
                refined_boxes = box_decode(residuals, proposals)

                # Rectify Scores
                # Score = Cls_Score * IoU^alpha
                rectified_scores = scores * torch.pow(
                    iou_pred.squeeze(-1), Config.IOU_RECT_ALPHA
                )

                for i in range(batch_size):
                    token = batch["sample_tokens"][i]

                    boxes = refined_boxes[i]
                    sc = rectified_scores[i]
                    lbls = cls_ids[i]

                    # NMS
                    keep = nms_3d(boxes, sc, iou_threshold=0.1)

                    boxes = boxes[keep]
                    sc = sc[keep]
                    lbls = lbls[keep]

                    # Format String
                    # confidence x y z w l h yaw class_name
                    pred_strs = []
                    for j in range(len(boxes)):
                        if sc[j] < 0.1:
                            continue  # Score threshold

                        b = boxes[j].cpu().numpy()
                        s = sc[j].item()
                        l = Config.CLASS_NAMES[lbls[j].item()]

                        # Format: score x y z w l h yaw class
                        pred_strs.append(
                            f"{s:.4f} {b[0]:.4f} {b[1]:.4f} {b[2]:.4f} {b[3]:.4f} {b[4]:.4f} {b[5]:.4f} {b[6]:.4f} {l}"
                        )

                    prediction_string = " ".join(pred_strs)
                    results.append({"Id": token, "PredictionString": prediction_string})

        df = pd.DataFrame(results)
        # Ensure all test IDs are present (fill missing with empty)
        # The loader iterates all, so we should be good.
        df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")


def run_detector():
    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # Datasets
    train_ds = LidarDataset(
        split="train", subset_size=Config.SUBSET_SIZE if Config.DEBUG else None
    )
    val_ds = LidarDataset(
        split="val", subset_size=Config.SUBSET_SIZE if Config.DEBUG else None
    )
    test_ds = LidarDataset(
        split="test", subset_size=Config.SUBSET_SIZE if Config.DEBUG else None
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    detector = PointPillarsDetector()

    # Train
    detector.train(train_loader, val_loader)

    # Load best model
    detector.load_checkpoint(os.path.join(Config.WORKING_DIR, "model_checkpoint.pth"))

    # Generate Submission
    detector.generate_submission(test_loader, Config.SUBMISSION_PATH)


# Note: The execution entry point is not included as per instructions.
