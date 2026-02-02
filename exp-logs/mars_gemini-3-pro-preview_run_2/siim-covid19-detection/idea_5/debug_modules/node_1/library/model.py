import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict

from library.config import (
    NUM_DETECTION_CLASSES,
    NUM_STUDY_CLASSES,
    RPN_PRE_NMS_TOP_N_TRAIN,
    RPN_POST_NMS_TOP_N_TRAIN,
    RPN_PRE_NMS_TOP_N_TEST,
    RPN_POST_NMS_TOP_N_TEST,
    BOX_DETECTIONS_PER_IMG,
    BOX_SCORE_THRESH,
    BOX_NMS_THRESH,
    MIL_POOL_SIZE,
    MIL_LOSS_WEIGHT,
    DEVICE,
)


class MILAttention(nn.Module):
    """
    Multi-Instance Learning Head with Gated Attention.
    Aggregates a bag of instance features into a single study-level prediction.
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MILAttention, self).__init__()
        self.L = input_dim
        self.D = hidden_dim
        self.K = num_classes

        self.attention_v = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())

        self.attention_u = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())

        self.attention_weights = nn.Linear(self.D, 1)
        self.classifier = nn.Linear(self.L, self.K)

    def forward(self, x):
        """
        Args:
            x: (Batch, Instances, Features)
        Returns:
            logits: (Batch, Num_Classes)
            A: Attention weights (Batch, Instances, 1)
        """
        # Gated Attention Mechanism
        # x: [B, K, L]

        a_v = self.attention_v(x)  # [B, K, D]
        a_u = self.attention_u(x)  # [B, K, D]

        # Element-wise multiplication (Gating)
        a = self.attention_weights(a_v * a_u)  # [B, K, 1]

        # Softmax over instances
        A = torch.softmax(a, dim=1)

        # Weighted sum of instances
        # [B, K, 1]^T * [B, K, L] -> [B, 1, L] -> [B, L]
        M = torch.sum(A * x, dim=1)

        logits = self.classifier(M)

        return logits, A


class InstanceAwareFasterRCNN(FasterRCNN):
    """
    Faster R-CNN with ResNet-101-FPN backbone and an auxiliary
    Multi-Instance Learning (MIL) head for study-level classification.
    """

    def __init__(self):
        # 1. Backbone: ResNet-101 FPN
        # We assume weights are available or fallback to pretrained=True logic internally handled by torchvision
        # Using standard resnet_fpn_backbone helper
        try:
            # Try modern torchvision API
            from torchvision.models import ResNet101_Weights

            backbone = resnet_fpn_backbone(
                "resnet101", weights=ResNet101_Weights.DEFAULT
            )
        except (ImportError, AttributeError):
            # Fallback for older versions
            backbone = resnet_fpn_backbone("resnet101", pretrained=True)

        # 2. RPN Configuration (High Capacity)
        rpn_anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        # 3. ROI Heads Configuration
        # We use the default MultiScaleRoIAlign
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        )

        # Initialize FasterRCNN
        super().__init__(
            backbone,
            num_classes=NUM_DETECTION_CLASSES,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler,
            # RPN Settings
            rpn_pre_nms_top_n_train=RPN_PRE_NMS_TOP_N_TRAIN,
            rpn_post_nms_top_n_train=RPN_POST_NMS_TOP_N_TRAIN,
            rpn_pre_nms_top_n_test=RPN_PRE_NMS_TOP_N_TEST,
            rpn_post_nms_top_n_test=RPN_POST_NMS_TOP_N_TEST,
            # Box Head Settings
            box_detections_per_img=BOX_DETECTIONS_PER_IMG,
            box_score_thresh=BOX_SCORE_THRESH,
            box_nms_thresh=BOX_NMS_THRESH,
        )

        # 4. MIL Head Setup
        # The box_head in FasterRCNN (TwoMLPHead) usually outputs 1024 dimensions
        representation_size = 1024
        self.mil_head = MILAttention(
            input_dim=representation_size, hidden_dim=512, num_classes=NUM_STUDY_CLASSES
        )

        self.mil_loss_weight = MIL_LOSS_WEIGHT

    def forward(self, images, targets=None):
        """
        Custom forward pass to handle MIL branch.
        """
        # 1. Standard Preprocessing (Transform)
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                    )
                else:
                    torch._assert(
                        False,
                        f"Expected target boxes to be of type Tensor, got {type(boxes)}.",
                    )

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last 2 dimensions of the tensor to be H and W. instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # 2. Backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # 3. RPN
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 4. ROI Heads (Detection Branch)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        # 5. MIL Branch (Study Classification)
        # We need to extract features for the top K proposals

        # Select Top K proposals per image for MIL
        mil_proposals = []
        for p in proposals:
            # Proposals are already sorted by score in RPN output
            # Take top K. If fewer than K, take all.
            k = min(len(p), MIL_POOL_SIZE)
            mil_proposals.append(p[:k])

        # Extract features using the existing ROI heads components
        # box_roi_pool expects a list of tensors and features dict
        # output: (Total_Instances, C, H, W)
        mil_box_features = self.roi_heads.box_roi_pool(
            features, mil_proposals, images.image_sizes
        )

        # Pass through the box head (TwoMLPHead) to get vector representation
        # output: (Total_Instances, 1024)
        mil_box_features = self.roi_heads.box_head(mil_box_features)

        # Reshape back to (Batch, K, Dim) for MIL Head
        # We need to handle variable K if some images had fewer proposals,
        # but with RPN_POST_NMS_TOP_N_TEST=1000, we assume we always have MIL_POOL_SIZE (64).
        # To be safe, we reconstruct the batch.

        split_sizes = [len(p) for p in mil_proposals]
        mil_box_features_split = torch.split(mil_box_features, split_sizes, dim=0)

        # Pad if necessary (though unlikely given config), and stack
        # For simplicity in this implementation, we assume we hit MIL_POOL_SIZE or handle the variable size in loop
        # The MIL Attention handles variable sequence length if we mask, but here we just process per batch item

        mil_logits_list = []

        for i, feats in enumerate(mil_box_features_split):
            # feats: (K_i, 1024)
            # Add batch dim: (1, K_i, 1024)
            feats_expanded = feats.unsqueeze(0)
            logits, _ = self.mil_head(feats_expanded)
            mil_logits_list.append(logits)

        mil_logits = torch.cat(mil_logits_list, dim=0)  # (Batch, Num_Study_Classes)

        # 6. Outputs & Losses
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)

        if self.training:
            # Calculate MIL Loss
            # targets contains 'study_label'
            study_labels = torch.stack([t["study_label"] for t in targets])
            loss_mil = F.cross_entropy(mil_logits, study_labels)
            losses["loss_mil"] = loss_mil * self.mil_loss_weight
            return losses
        else:
            # Inference Mode
            # Attach study predictions to the detection results
            # detections is a list of dicts

            study_probs = F.softmax(mil_logits, dim=1)

            for i, det in enumerate(detections):
                det["study_logits"] = mil_logits[i]
                det["study_probs"] = study_probs[i]

            return detections


def get_model():
    """
    Factory function to create the model and move it to the configured device.
    """
    model = InstanceAwareFasterRCNN()
    model.to(DEVICE)
    return model
