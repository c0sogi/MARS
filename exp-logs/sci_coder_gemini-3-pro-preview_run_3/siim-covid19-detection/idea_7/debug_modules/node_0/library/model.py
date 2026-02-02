import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RPNHead,
    RegionProposalNetwork,
)
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.box_coder import BoxCoder
from torchvision.models.detection.proposal_matcher import ProposalMatcher
from torchvision.models.detection.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler,
)
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from library.config import Config

# ====================================================
# Helper Modules
# ====================================================


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class AttentionPoolingHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_heads=8, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads

        # Positional encoding can be learned or fixed. Using learned for simplicity.
        # Assuming max feature map size (e.g., 25x25 for 800px image at stride 32)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, in_channels) * 0.02)

        self.norm = nn.LayerNorm(in_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Flatten spatial dimensions: (B, C, H*W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)

        # Add positional embedding (interpolate if size mismatch)
        if x.shape[1] > self.pos_embed.shape[1]:
            # This case shouldn't happen with fixed size, but for safety
            x = x[:, : self.pos_embed.shape[1], :]

        pos = self.pos_embed[:, : x.shape[1], :]
        x = x + pos

        x = self.norm(x)

        # Self-Attention
        # attn_output: (B, L, C)
        attn_output, _ = self.attention(x, x, x)

        # Global Average Pooling over the sequence
        x = attn_output.mean(dim=1)  # (B, C)

        # Classification
        logits = self.fc(x)
        return logits


# ====================================================
# Cascade RoI Heads
# ====================================================


class CascadeRoIHeads(nn.Module):
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        fg_iou_threshs,
        bg_iou_threshs,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        score_thresh,
        nms_thresh,
        detections_per_img,
    ):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor  # List of predictors

        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.detections_per_img = detections_per_img

        self.stages = len(fg_iou_threshs)
        self.proposal_matcher = []
        self.box_coder = []
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        for i in range(self.stages):
            # Matcher for each stage
            self.proposal_matcher.append(
                ProposalMatcher(
                    fg_iou_threshs[i],
                    bg_iou_threshs[i],
                    allow_low_quality_matches=False,
                )
            )
            # Box Coder for each stage (weights usually increase)
            # Standard Cascade: [10, 10, 5, 5], [20, 20, 10, 10], [30, 30, 15, 15]
            weights = bbox_reg_weights[i]
            self.box_coder.append(BoxCoder(weights=weights))

        self.proposal_matcher = nn.ModuleList(self.proposal_matcher)
        # BoxCoder is not an nn.Module, just a class

    def check_targets(self, targets):
        if targets is None:
            raise ValueError("targets should not be None")
        if not all(["boxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a boxes key")
        if not all(["labels" in t for t in targets]):
            raise ValueError("Every element of targets should have a labels key")

    def select_training_samples(self, proposals, targets, stage_idx):
        self.check_targets(targets)
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposals
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        # get matching gt indices for each proposal
        matched_idxs, labels = self.proposal_matcher[stage_idx](gt_boxes, proposals)

        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.fg_bg_sampler(labels)

        sampled_proposals = []
        matched_gt_boxes = []
        num_images = len(proposals)

        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals_per_image = proposals[img_id][img_sampled_inds]
            gt_boxes_per_image = gt_boxes[img_id]
            matched_idxs_per_image = matched_idxs[img_id][img_sampled_inds]

            sampled_proposals.append(proposals_per_image)
            matched_gt_boxes.append(gt_boxes_per_image[matched_idxs_per_image])

        # get labels for sampled proposals
        # Clamping matched_idxs to avoid error for bg (which is -1, but we want to index labels)
        # Background labels are handled by the matcher setting label to 0 where match is low
        # But here 'labels' variable contains 0 for bg, 1 for ignore, positive for fg.
        # We need actual class labels.

        sampled_labels = []
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            # labels from matcher: 1=pos, 0=bg, -1=ignore
            matcher_labels = labels[img_id][img_sampled_inds]

            # Real class labels
            gt_labels_per_image = gt_labels[img_id]
            matched_idxs_per_image = matched_idxs[img_id][img_sampled_inds]

            # Create final labels tensor
            # Initialize with 0 (background)
            final_labels = torch.zeros_like(matcher_labels, dtype=torch.int64)

            # Fill positive labels
            pos_mask = matcher_labels > 0
            if pos_mask.any():
                final_labels[pos_mask] = gt_labels_per_image[
                    matched_idxs_per_image[pos_mask]
                ]

            sampled_labels.append(final_labels)

        # compute regression targets
        regression_targets = self.box_coder[stage_idx].encode(
            matched_gt_boxes, sampled_proposals
        )

        return sampled_proposals, sampled_labels, regression_targets

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Args:
            features (OrderedDict): feature maps
            proposals (List[Tensor]): proposals from RPN
            image_shapes (List[Tuple[int, int]]): shapes of images
            targets (List[Dict]): ground truth
        """
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(
                        f"target boxes must of float type, instead got {t['boxes'].dtype}"
                    )
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(
                        f"target labels must of int64 type, instead got {t['labels'].dtype}"
                    )

        losses = {}
        all_detections = []

        # Initial proposals from RPN
        curr_proposals = proposals

        for i in range(self.stages):
            # 1. Sampling (Training only)
            if self.training:
                proposals_sampled, labels, regression_targets = (
                    self.select_training_samples(curr_proposals, targets, i)
                )
            else:
                proposals_sampled = curr_proposals
                labels = None
                regression_targets = None

            # 2. RoI Align
            box_features = self.box_roi_pool(features, proposals_sampled, image_shapes)

            # 3. Head & Predictor
            # Shared head usually, separate predictors
            box_features = self.box_head(box_features)
            class_logits, box_regression = self.box_predictor[i](box_features)

            # 4. Loss (Training only)
            if self.training:
                loss_classifier, loss_box_reg = fastrcnn_loss(
                    class_logits, box_regression, labels, regression_targets
                )
                losses[f"loss_classifier_s{i}"] = loss_classifier
                losses[f"loss_box_reg_s{i}"] = loss_box_reg

            # 5. Refine Boxes (Decode) for next stage or final output
            result = []
            boxes, scores, pred_labels = self.postprocess_detections(
                class_logits, box_regression, proposals_sampled, image_shapes, i
            )

            # Prepare proposals for next stage
            # We use the refined boxes as proposals for the next stage
            # Detach to stop gradient backprop to previous stage via boxes
            next_proposals = [b.detach() for b in boxes]
            curr_proposals = next_proposals

            # Store detections
            all_detections.append((boxes, scores, pred_labels))

        if self.training:
            return losses, None
        else:
            # Inference: Use the output from the last stage
            # Or average scores. For simplicity and standard behavior in many implementations,
            # we use the refined boxes from the last stage and the scores from the last stage.
            boxes, scores, labels = all_detections[-1]
            return {}, [
                {"boxes": b, "scores": s, "labels": l}
                for b, s, l in zip(boxes, scores, labels)
            ]

    def postprocess_detections(
        self, class_logits, box_regression, proposals, image_shapes, stage_idx
    ):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder[stage_idx].decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []

        for boxes, scores, image_shape in zip(
            pred_boxes_list, pred_scores_list, image_shapes
        ):
            # Clip boxes to image
            boxes = torchvision.ops.clip_boxes_to_image(boxes, image_shape)

            # For intermediate stages, we might just want the boxes for the next stage
            # For the final stage (or if we want to return predictions), we do NMS

            # However, in Cascade R-CNN training loop, we typically keep all refined boxes
            # (or a subset) as proposals for the next stage without aggressive NMS
            # to maintain diversity, but usually we just pass them through.
            # Here we return all refined boxes.

            # If this is inference and final stage, we filter.
            # But the logic inside forward loop needs raw boxes for next stage.
            # So we return raw refined boxes here.

            # Create labels (just argmax for now, used for next stage proposal if needed?)
            # Actually next stage just needs boxes.

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(torch.argmax(scores, dim=1))  # Dummy labels

        # If it's the final stage and we are in inference, we need to apply NMS and formatting
        # But doing it here complicates the loop.
        # We will do final post-processing outside the loop or in a separate block in forward.

        # Wait, the forward method logic for inference needs to return the final clean predictions.
        # Let's refine the return of `forward` for inference.

        return all_boxes, all_scores, all_labels


# ====================================================
# Main Model Class
# ====================================================


class CovidCascadeRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Backbone (ConvNeXt V2 Base)
        # features_only=True returns a list of feature maps
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # Stride 4, 8, 16, 32
        )

        # Get channel counts
        dummy_input = torch.randn(1, 3, 256, 256)
        feats = self.backbone(dummy_input)
        in_channels_list = [f.shape[1] for f in feats]

        # 2. FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=dict(
                zip([str(i) for i in range(len(in_channels_list))], in_channels_list)
            ),
            out_channels=256,
        )

        # 3. RPN
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
        rpn_head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

        # Standard RPN config
        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=dict(
                training=Config.RPN_PRE_NMS_TOP_N_TRAIN,
                testing=Config.RPN_PRE_NMS_TOP_N_TEST,
            ),
            post_nms_top_n=dict(
                training=Config.RPN_POST_NMS_TOP_N_TRAIN,
                testing=Config.RPN_POST_NMS_TOP_N_TEST,
            ),
            nms_thresh=Config.RPN_NMS_THRESH,
        )

        # 4. Cascade RoI Heads
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        )

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(256 * resolution**2, representation_size)

        # 3 Predictors for 3 stages
        box_predictors = nn.ModuleList(
            [
                FastRCNNPredictor(representation_size, Config.NUM_DETECTION_CLASSES)
                for _ in range(3)
            ]
        )

        # Cascade Config
        fg_iou_threshs = Config.CASCADE_IOU_THRESHOLDS  # [0.5, 0.6, 0.7]
        bg_iou_threshs = [0.5, 0.6, 0.7]  # Usually bg is [0, thr)
        # Actually Matcher expects low and high.
        # For Cascade, we usually set thresholds such that we select samples > thr.
        # ProposalMatcher(fg_iou_thresh, bg_iou_thresh)
        # If we pass single values, it treats as high/low boundaries differently?
        # No, ProposalMatcher takes high_threshold and low_threshold.
        # Between low and high is ignore. Below low is bg. Above high is fg.
        # For Cascade stage k, we want samples with IoU > thr_k.
        # So low_threshold = thr_k.

        self.roi_heads = CascadeRoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictors,
            fg_iou_threshs=fg_iou_threshs,
            bg_iou_threshs=fg_iou_threshs,  # Set bg thresh same as fg to ignore 'between' or define strict split
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=[
                (10.0, 10.0, 5.0, 5.0),
                (20.0, 20.0, 10.0, 10.0),
                (30.0, 30.0, 15.0, 15.0),
            ],
            score_thresh=Config.BOX_SCORE_THRESH,
            nms_thresh=Config.BOX_NMS_THRESH,
            detections_per_img=Config.BOX_DETECTIONS_PER_IMG,
        )

        # 5. Study Classification Head (Attention)
        self.study_head = AttentionPoolingHead(
            in_channels=256, num_classes=Config.NUM_STUDY_CLASSES, num_heads=8
        )

        # Transform (mainly for batching/resizing if needed, but we handle resizing in dataset)
        # We use it to normalize if not done in dataset, but dataset does it.
        # We use it to wrap into ImageList.
        self.transform = GeneralizedRCNNTransform(
            min_size=Config.IMG_SIZE,
            max_size=Config.IMG_SIZE,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )
        # Note: Dataset already normalizes. So we should disable normalization here.
        # Setting mean/std to None makes it skip normalization.
        self.transform.image_mean = None
        self.transform.image_std = None

    def forward(self, images, targets=None):
        # 1. Transform (Batching & ImageList wrapping)
        # images is a tensor (B, C, H, W). GeneralizedRCNN expects list of tensors.
        if isinstance(images, torch.Tensor):
            images_list = [img for img in images]
        else:
            images_list = images

        # If training, targets must be provided
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        original_image_sizes = []
        for img in images_list:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images_list, targets)

        # 2. Backbone & FPN
        features_list = self.backbone(images.tensors)
        # Convert list to dict for FPN: '0', '1', '2', '3'
        features_dict = {str(i): f for i, f in enumerate(features_list)}
        features = self.fpn(features_dict)

        # 3. RPN
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 4. ROI Heads (Cascade)
        roi_losses, detections = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        # 5. Study Head
        # Use the deepest FPN layer (P5, key '3')
        deepest_feat = features["3"]
        study_logits = self.study_head(deepest_feat)

        losses = {}
        losses.update(proposal_losses)
        losses.update(roi_losses)

        if self.training:
            # Calculate Study Loss (Focal Loss)
            # targets is a list of dicts. Get study_label.
            study_targets = torch.stack([t["study_label"] for t in targets])

            # One-hot encode targets for Focal Loss
            study_targets_one_hot = F.one_hot(
                study_targets, num_classes=Config.NUM_STUDY_CLASSES
            ).float()

            loss_study = sigmoid_focal_loss(
                study_logits, study_targets_one_hot, reduction="mean"
            )
            losses["loss_study"] = loss_study * Config.LOSS_WEIGHT_STUDY

            return losses
        else:
            # Inference
            study_probs = torch.sigmoid(study_logits)

            # Post-process detections for final output
            # detections is a list of dicts: {'boxes': ..., 'scores': ..., 'labels': ...}
            # We need to apply NMS here since CascadeRoIHeads returned raw refined boxes from last stage

            final_detections = []
            for i, det in enumerate(detections):
                boxes = det["boxes"]
                scores = det["scores"]  # (N, num_classes)

                # Apply score threshold and NMS per class
                # We only have 1 class 'opacity' (index 1). Background is 0.

                # Get scores for class 1
                opacity_scores = scores[:, 1]

                # Filter by score
                keep = opacity_scores > Config.BOX_SCORE_THRESH
                boxes = boxes[keep]
                opacity_scores = opacity_scores[keep]
                labels = torch.ones_like(
                    opacity_scores, dtype=torch.int64
                )  # All are opacity

                # Apply NMS
                keep_nms = torchvision.ops.nms(
                    boxes, opacity_scores, Config.BOX_NMS_THRESH
                )

                # Limit detections
                keep_nms = keep_nms[: Config.BOX_DETECTIONS_PER_IMG]

                boxes = boxes[keep_nms]
                opacity_scores = opacity_scores[keep_nms]
                labels = labels[keep_nms]

                # Resize boxes back to original image size
                # GeneralizedRCNNTransform handles resizing, but we need to map back
                # The 'transform' object has a method 'postprocess' but it expects the full prediction structure

                # Manual resize back
                # Current image size: images.image_sizes[i]
                # Original size: original_image_sizes[i]

                curr_h, curr_w = images.image_sizes[i]
                orig_h, orig_w = original_image_sizes[i]

                scale_w = orig_w / curr_w
                scale_h = orig_h / curr_h

                boxes[:, 0] *= scale_w
                boxes[:, 2] *= scale_w
                boxes[:, 1] *= scale_h
                boxes[:, 3] *= scale_h

                final_detections.append(
                    {"boxes": boxes, "scores": opacity_scores, "labels": labels}
                )

            return final_detections, study_probs
