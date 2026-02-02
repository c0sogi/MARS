import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RPNHead,
    RegionProposalNetwork,
)
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import boxes as box_ops
import timm
from collections import OrderedDict
from library.config import Config


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C) logits
        # targets: (N) indices or (N, C) one-hot

        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class SwinTransformerBackbone(nn.Module):
    def __init__(self, model_name, out_indices=(0, 1, 2, 3)):
        super(SwinTransformerBackbone, self).__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=True, features_only=True, out_indices=out_indices
        )
        # Get output channels from the model config
        self.out_channels = self.backbone.feature_info.channels()

    def forward(self, x):
        return self.backbone(x)


class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        super(BackboneWithFPN, self).__init__()
        self.body = backbone
        self.return_layers = return_layers

        # FPN
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=torchvision.ops.LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        # Convert list of tensors to OrderedDict for FPN
        inputs = OrderedDict()
        for i, name in enumerate(self.return_layers):
            inputs[name] = x[i]

        out = self.fpn(inputs)
        return out


class StudyClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(StudyClassifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x is a feature map (N, C, H, W)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CascadeRoIHeads(nn.Module):
    def __init__(self, in_channels, num_classes, iou_thresholds=[0.5, 0.6, 0.7]):
        super(CascadeRoIHeads, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds

        # ROI Align
        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        )

        # Heads and Predictors for each cascade stage
        self.box_heads = nn.ModuleList()
        self.box_predictors = nn.ModuleList()

        for _ in range(len(iou_thresholds)):
            # Standard TwoMLPHead
            head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
                in_channels * 7 * 7, 1024
            )
            # Standard Predictor
            predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                1024, num_classes
            )
            self.box_heads.append(head)
            self.box_predictors.append(predictor)

        # Proposal Matcher and Sampler components
        # We will instantiate these dynamically or use helper functions,
        # but to keep it clean we define the logic in forward.
        self.fg_bg_sampler = (
            torchvision.models.detection.roi_heads.BalancedPositiveNegativeSampler(
                batch_size_per_image=512, positive_fraction=0.25
            )
        )

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, iou_thresh):
        matched_idxs = []
        labels = []

        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(
            proposals, gt_boxes, gt_labels
        ):
            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                match_quality_matrix = box_ops.box_iou(
                    gt_boxes_in_image, proposals_in_image
                )

                # Matcher logic (simplified High-IoU matcher)
                matched_vals, matches = match_quality_matrix.max(dim=0)

                # Assign background (0) if below threshold
                # Assign foreground (original label) if above threshold
                # We use a trick: matches points to GT index.

                below_low_threshold = matched_vals < iou_thresh
                between_thresholds = (
                    matched_vals >= iou_thresh
                )  # & (matched_vals < high) - strictly we just want > thresh for cascade

                # For Cascade, we treat everything < thresh as background for this stage

                clamped_matched_idxs_in_image = matches.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Set background label (0)
                labels_in_image[below_low_threshold] = 0

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)

        return matched_idxs, labels

    def subsample(self, labels):
        # Uses BalancedPositiveNegativeSampler
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for the
        # corresponding ground truth labels, to be used with advanced indexing
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Args:
            features (Dict[str, Tensor]): feature pyramid
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
        final_proposals = proposals
        final_class_logits = None
        final_box_regression = None

        # Cascade Loop
        for i, (iou_thresh, box_head, box_predictor) in enumerate(
            zip(self.iou_thresholds, self.box_heads, self.box_predictors)
        ):

            stage_proposals = final_proposals

            if self.training:
                # 1. Match and Sample
                gt_boxes = [t["boxes"] for t in targets]
                gt_labels = [t["labels"] for t in targets]

                matched_idxs, labels = self.assign_targets_to_proposals(
                    stage_proposals, gt_boxes, gt_labels, iou_thresh
                )
                sampled_inds = self.subsample(labels)

                # Filter proposals, matched_idxs, labels based on sampling
                sampled_proposals = []
                sampled_labels = []
                sampled_regression_targets = []

                for img_idx, inds in enumerate(sampled_inds):
                    img_proposals = stage_proposals[img_idx][inds]
                    img_labels = labels[img_idx][inds]
                    sampled_proposals.append(img_proposals)
                    sampled_labels.append(img_labels)

                    # Compute regression targets
                    # matched_idxs[img_idx][inds] points to the GT box for each sampled proposal
                    # If label is 0 (bg), regression target doesn't matter much but we calculate it
                    gt_boxes_in_image = gt_boxes[img_idx]
                    if gt_boxes_in_image.numel() > 0:
                        matched_gt_boxes = gt_boxes_in_image[
                            matched_idxs[img_idx][inds]
                        ]
                        reg_targets = torchvision.ops.boxes.box_coder.encode_boxes(
                            matched_gt_boxes,
                            img_proposals,
                            weights=(10.0, 10.0, 5.0, 5.0),
                        )
                    else:
                        reg_targets = torch.zeros_like(img_proposals)

                    sampled_regression_targets.append(reg_targets)

                stage_proposals_to_pool = sampled_proposals
            else:
                stage_proposals_to_pool = stage_proposals

            # 2. ROI Align
            box_features = self.box_roi_pool(
                features, stage_proposals_to_pool, image_shapes
            )
            box_features = box_head(box_features)
            class_logits, box_regression = box_predictor(box_features)

            if self.training:
                # 3. Compute Loss
                loss_cls, loss_box = self.fastrcnn_loss(
                    class_logits,
                    box_regression,
                    sampled_labels,
                    sampled_regression_targets,
                )
                losses[f"loss_cascade_cls_{i}"] = loss_cls * Config.LOSS_WEIGHT_ROI_CLS
                losses[f"loss_cascade_box_{i}"] = loss_box * Config.LOSS_WEIGHT_ROI_BOX

                # 4. Refine Proposals for next stage (if not last stage)
                if i < len(self.iou_thresholds) - 1:
                    with torch.no_grad():
                        # Decode boxes
                        # We need to apply regression to the *original* proposals (stage_proposals_to_pool)
                        # box_regression shape: (N, num_classes * 4)
                        # We use the regression corresponding to the predicted class or just class-agnostic?
                        # Usually Cascade R-CNN uses class agnostic or specific.
                        # To simplify, we use the regression for the assigned label (during training)
                        # or the max score label.

                        # For refinement during training, we use the regression corresponding to the GT label
                        # But wait, we need proposals for the next stage which might include Backgrounds?
                        # Actually, usually we only refine positive proposals or all.
                        # Standard Cascade: Refine all.

                        # Let's use class-agnostic decoding for simplicity or just take the one with highest score.
                        # However, during training we know the labels.

                        # Simplified refinement: Use the regression corresponding to the predicted class?
                        # No, simpler: Just use the regression for class 1 (opacity) if we assume binary+bg,
                        # or use the regression for the specific class.

                        # Let's assume we refine using the regression of the predicted class.
                        pred_scores = F.softmax(class_logits, dim=-1)
                        pred_labels = torch.argmax(pred_scores, dim=1)

                        # Expand proposals to match box_regression shape logic
                        proposals_tensor = torch.cat(stage_proposals_to_pool, dim=0)

                        # Decode
                        boxes_per_image = [len(p) for p in stage_proposals_to_pool]

                        pred_boxes = torchvision.ops.boxes.box_coder.decode_boxes(
                            box_regression,
                            proposals_tensor,
                            weights=(10.0, 10.0, 5.0, 5.0),
                        )

                        # Select the box corresponding to the predicted label
                        # pred_boxes: (N, C, 4)
                        # We select (N, 4) based on pred_labels
                        # But we must ensure we don't select background regression if we want to find objects?
                        # Actually, we just want to refine the box.

                        # For simplicity in this implementation:
                        # Just take the regression for class 1 (Opacity) for all,
                        # assuming we are looking for opacities.
                        # Or better: take the regression for the class with highest score.

                        N_boxes = pred_boxes.shape[0]
                        pred_boxes = pred_boxes.reshape(N_boxes, -1, 4)
                        # Gather
                        refined_boxes = pred_boxes[torch.arange(N_boxes), pred_labels]

                        # Clip to image
                        # We need to split back to list to clip per image
                        refined_proposals_list = refined_boxes.split(boxes_per_image)
                        new_proposals = []
                        for img_idx, (props, shape) in enumerate(
                            zip(refined_proposals_list, image_shapes)
                        ):
                            props = box_ops.clip_boxes_to_image(props, shape)
                            new_proposals.append(props)

                        final_proposals = new_proposals
            else:
                # Inference
                final_class_logits = class_logits
                final_box_regression = box_regression

                if i < len(self.iou_thresholds) - 1:
                    # Refine for next stage
                    pred_scores = F.softmax(class_logits, dim=-1)
                    pred_labels = torch.argmax(pred_scores, dim=1)
                    proposals_tensor = torch.cat(stage_proposals_to_pool, dim=0)
                    boxes_per_image = [len(p) for p in stage_proposals_to_pool]

                    pred_boxes = torchvision.ops.boxes.box_coder.decode_boxes(
                        box_regression, proposals_tensor, weights=(10.0, 10.0, 5.0, 5.0)
                    )
                    pred_boxes = pred_boxes.reshape(pred_boxes.shape[0], -1, 4)
                    refined_boxes = pred_boxes[
                        torch.arange(pred_boxes.shape[0]), pred_labels
                    ]

                    refined_proposals_list = refined_boxes.split(boxes_per_image)
                    new_proposals = []
                    for img_idx, (props, shape) in enumerate(
                        zip(refined_proposals_list, image_shapes)
                    ):
                        props = box_ops.clip_boxes_to_image(props, shape)
                        new_proposals.append(props)
                    final_proposals = new_proposals

        if self.training:
            return losses
        else:
            # Post-process final stage
            boxes, scores, labels = self.postprocess_detections(
                final_class_logits, final_box_regression, final_proposals, image_shapes
            )
            return [
                {"boxes": b, "labels": l, "scores": s}
                for b, l, s in zip(boxes, labels, scores)
            ]

    def postprocess_detections(
        self, class_logits, box_regression, proposals, image_shapes
    ):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = torchvision.ops.boxes.box_coder.decode_boxes(
            box_regression, torch.cat(proposals, dim=0), weights=(10.0, 10.0, 5.0, 5.0)
        )
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []

        for boxes, scores, image_shape in zip(
            pred_boxes_list, pred_scores_list, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # Create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # Remove background class (0)
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # Batch everything
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # Remove low scoring boxes
            inds = torch.where(scores > Config.CONF_THRESHOLD)[0]
            boxes = boxes[inds]
            scores = scores[inds]
            labels = labels[inds]

            # NMS
            keep = box_ops.nms(boxes, scores, Config.NMS_IOU_THRESHOLD)
            all_boxes.append(boxes[keep])
            all_scores.append(scores[keep])
            all_labels.append(labels[keep])

        return all_boxes, all_scores, all_labels


class SwinCascadeRCNN(nn.Module):
    def __init__(self):
        super(SwinCascadeRCNN, self).__init__()

        # 1. Backbone
        backbone = SwinTransformerBackbone(
            Config.BACKBONE_NAME, out_indices=(0, 1, 2, 3)
        )
        backbone_out_channels = Config.BACKBONE_OUT_CHANNELS

        # 2. FPN
        self.backbone_fpn = BackboneWithFPN(
            backbone,
            return_layers=["0", "1", "2", "3"],
            in_channels_list=backbone_out_channels,
            out_channels=Config.FPN_OUT_CHANNELS,
        )

        # 3. RPN
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
        rpn_head = RPNHead(
            Config.FPN_OUT_CHANNELS, anchor_generator.num_anchors_per_location()[0]
        )

        # We manually manage RPN to have full control
        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.7,
        )

        # 4. Cascade RoI Heads
        self.roi_heads = CascadeRoIHeads(
            in_channels=Config.FPN_OUT_CHANNELS,
            num_classes=Config.NUM_CLASSES_DETECTION,
            iou_thresholds=Config.CASCADE_IOU_THRESHOLDS,
        )

        # 5. Study Classifier
        self.study_classifier = StudyClassifier(
            in_channels=Config.FPN_OUT_CHANNELS, num_classes=Config.NUM_CLASSES_STUDY
        )

        # Loss for study
        self.study_criterion = FocalLoss(
            alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA
        )

    def forward(self, images, targets=None):
        """
        Args:
            images: List[Tensor] or Tensor (B, C, H, W)
            targets: List[Dict]
        """
        # Wrap images in ImageList
        if isinstance(images, torch.Tensor):
            # Assuming images are already resized to same size by collate_fn
            img_list = ImageList(images, [img.shape[-2:] for img in images])
        else:
            # List of tensors
            img_list = ImageList(
                torch.stack(images), [img.shape[-2:] for img in images]
            )

        # Feature Extraction
        features = self.backbone_fpn(img_list.tensors)

        # RPN
        proposals, proposal_losses = self.rpn(img_list, features, targets)

        # Cascade RoI Heads
        detections_or_losses = self.roi_heads(
            features, proposals, img_list.image_sizes, targets
        )

        # Study Classification
        # Use the deepest feature map (P5) corresponding to '3'
        # '0'->P2, '1'->P3, '2'->P4, '3'->P5
        study_features = features["3"]
        study_logits = self.study_classifier(study_features)

        if self.training:
            losses = {}
            losses.update(proposal_losses)
            losses.update(detections_or_losses)

            # Calculate Study Loss
            study_targets = torch.stack([t["study_label"] for t in targets])
            study_loss = self.study_criterion(study_logits, study_targets)
            losses["loss_study"] = study_loss * Config.LOSS_WEIGHT_STUDY

            return losses
        else:
            # Inference
            # detections_or_losses is a list of dicts {'boxes', 'labels', 'scores'}

            # Add study predictions
            study_probs = torch.softmax(study_logits, dim=1)

            results = []
            for i, det in enumerate(detections_or_losses):
                res = det.copy()
                res["study_probs"] = study_probs[i]
                results.append(res)

            return results
