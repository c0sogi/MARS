import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.modules import (
    PillarVFE,
    PointPillarsScatter,
    BEVBackbone,
    CenterHead,
    RoIHead,
)
from library.utils import nms_3d, decode_refinement


class TwoStagePointPillars(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Feature Encoder
        self.vfe = PillarVFE()

        # 2. Scatter to BEV
        self.scatter = PointPillarsScatter()

        # 3. Backbone (ResNet-FPN)
        self.backbone = BEVBackbone()

        # Calculate backbone output channels
        # Config.NUM_UPSAMPLE_FILTERS = [128, 128, 128] -> 384
        c_out = sum(Config.NUM_UPSAMPLE_FILTERS)

        # 4. Stage 1: Proposal Head (CenterPoint-like)
        self.center_head = CenterHead(c_out)

        # 5. Stage 2: Refinement Head
        self.roi_head = RoIHead(c_out)

    def get_proposals_corrected(self, hm, reg, k=Config.PRE_MAX_SIZE):
        """
        Decodes heatmap and regression maps into 3D boxes with class labels.
        Corrects the class decoding logic missing in the library implementation.
        """
        batch, cat, height, width = hm.size()

        # Flatten heatmap: (B, C * H * W)
        hm_flat = hm.view(batch, -1)
        scores, inds = torch.topk(hm_flat, k)

        # Decode indices
        # inds = class_idx * (H*W) + pixel_idx
        class_ids = (inds // (height * width)).float()
        pixel_inds = inds % (height * width)
        y_inds = (pixel_inds // width).float()
        x_inds = (pixel_inds % width).float()

        # Gather regression targets
        # reg: (B, 8, H, W) -> (B, 8, H*W)
        reg_flat = reg.view(batch, 8, -1)

        # Expand pixel_inds for gathering: (B, 8, K)
        pixel_inds_expanded = pixel_inds.unsqueeze(1).expand(-1, 8, -1)

        # Gather: (B, 8, K) -> (B, K, 8)
        reg_vals = torch.gather(reg_flat, 2, pixel_inds_expanded).permute(0, 2, 1)

        # Decode Box Parameters
        ox, oy, z = reg_vals[..., 0], reg_vals[..., 1], reg_vals[..., 2]
        lw, ll, lh = reg_vals[..., 3], reg_vals[..., 4], reg_vals[..., 5]
        sin, cos = reg_vals[..., 6], reg_vals[..., 7]

        # Grid to World Coordinates
        stride_x = Config.VOXEL_SIZE[0] * Config.FEATURE_MAP_STRIDE
        stride_y = Config.VOXEL_SIZE[1] * Config.FEATURE_MAP_STRIDE

        xs = x_inds + ox
        ys = y_inds + oy

        x_world = xs * stride_x + Config.POINT_CLOUD_RANGE[0]
        y_world = ys * stride_y + Config.POINT_CLOUD_RANGE[1]

        # Dimensions and Yaw
        w = torch.exp(lw)
        l = torch.exp(ll)
        h = torch.exp(lh)
        yaw = torch.atan2(sin, cos)

        # Stack: (B, K, 7) [x, y, z, w, l, h, yaw]
        boxes = torch.stack([x_world, y_world, z, w, l, h, yaw], dim=-1)

        return boxes, scores, class_ids

    def forward(self, data_dict, mode="train"):
        """
        Forward pass for training or inference.
        """
        voxels = data_dict["voxels"]
        num_points = data_dict["num_points"]
        coords = data_dict["coordinates"]

        # 1. VFE: (M, MaxPoints, 9) -> (M, 64)
        pillar_feats = self.vfe(voxels, num_points)

        # 2. Scatter: (M, 64) -> (B, 64, H_in, W_in)
        batch_size = coords[:, 0].max().item() + 1
        bev_map = self.scatter(pillar_feats, coords, batch_size)

        # 3. Backbone: (B, 64, H_in, W_in) -> (B, 384, H_out, W_out)
        feature_map = self.backbone(bev_map)

        # 4. Stage 1 Head
        hm, reg = self.center_head(feature_map)

        if mode == "train":
            gt_boxes = data_dict["gt_boxes"]

            # Stage 1 Loss
            loss_hm, loss_box = self.center_head.loss(hm, reg, gt_boxes)

            # Generate Proposals for Stage 2 (No Gradients)
            with torch.no_grad():
                proposals, _, _ = self.get_proposals_corrected(hm, reg, k=200)

            # Stage 2 Loss (Refinement)
            loss_refine = self.roi_head.loss(feature_map, proposals, gt_boxes)

            return {
                "loss_hm": loss_hm,
                "loss_box": loss_box,
                "loss_refine": loss_refine,
                "total_loss": loss_hm + loss_box + loss_refine,
            }

        else:
            # Inference Pipeline
            proposals, scores, labels = self.get_proposals_corrected(
                hm, reg, k=Config.PRE_MAX_SIZE
            )

            final_boxes_list = []
            final_scores_list = []
            final_labels_list = []

            for b in range(batch_size):
                b_boxes = proposals[b]
                b_scores = scores[b]
                b_labels = labels[b]

                # 1. Score Thresholding
                mask = b_scores > Config.SCORE_THRESHOLD
                b_boxes = b_boxes[mask]
                b_scores = b_scores[mask]
                b_labels = b_labels[mask]

                if len(b_boxes) > 0:
                    # 2. NMS (Stage 1)
                    keep = nms_3d(b_boxes, b_scores, threshold=Config.NMS_IOU_THRESHOLD)

                    # Limit number of boxes
                    keep = keep[: Config.POST_MAX_SIZE]

                    b_boxes = b_boxes[keep]
                    b_scores = b_scores[keep]
                    b_labels = b_labels[keep]

                    # 3. Stage 2 Refinement
                    # RoIHead expects (B, N, 7) and (B, C, H, W)
                    # We process single sample, so unsqueeze batch dim
                    roi_boxes_in = b_boxes.unsqueeze(0)
                    roi_feats_in = feature_map[b].unsqueeze(0)

                    # Predict Residuals
                    residuals = self.roi_head(roi_feats_in, roi_boxes_in)

                    # Decode Final Boxes
                    refined_boxes = decode_refinement(roi_boxes_in, residuals)

                    final_boxes_list.append(refined_boxes[0])
                    final_scores_list.append(b_scores)
                    final_labels_list.append(b_labels)
                else:
                    # Handle empty predictions
                    final_boxes_list.append(torch.zeros((0, 7), device=hm.device))
                    final_scores_list.append(torch.zeros((0,), device=hm.device))
                    final_labels_list.append(torch.zeros((0,), device=hm.device))

            return final_boxes_list, final_scores_list, final_labels_list
