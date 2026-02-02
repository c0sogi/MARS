import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from collections import OrderedDict
from library.config import Config


class MultiTaskFasterRCNN(FasterRCNN):
    def __init__(self):
        """
        Initializes the Multi-Task Faster R-CNN with ResNet101-FPN backbone
        and an auxiliary Global Classification Head.
        """
        # 1. Backbone: ResNet101-FPN
        # We use 'DEFAULT' weights (ImageNet) and make the top 3 blocks trainable
        backbone = resnet_fpn_backbone(
            "resnet101", weights="DEFAULT", trainable_layers=3
        )

        # 2. Initialize Parent FasterRCNN
        # Note: We set image_mean=[0,0,0] and image_std=[1,1,1] because the
        # SIIMDataset already applies normalization (Albumentations).
        # We rely on the internal transform only for resizing and batching.
        super().__init__(
            backbone=backbone,
            num_classes=Config.NUM_DETECTION_CLASSES,
            min_size=Config.IMG_SIZE,
            max_size=Config.IMG_SIZE,
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            rpn_pre_nms_top_n_train=Config.RPN_PRE_NMS_TOP_N_TRAIN,
            rpn_post_nms_top_n_train=Config.RPN_POST_NMS_TOP_N_TRAIN,
            rpn_pre_nms_top_n_test=Config.RPN_PRE_NMS_TOP_N_TEST,
            rpn_post_nms_top_n_test=Config.RPN_POST_NMS_TOP_N_TEST,
        )

        # 3. Global Classification Head
        # The FPN backbone outputs 256 channels.
        # We will pool the feature maps and project to the study classes.
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, Config.NUM_STUDY_CLASSES),
        )

        self.lambda_global = Config.LAMBDA_GLOBAL_CLS
        self.study_loss_fn = nn.CrossEntropyLoss()

    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor]): List of images.
            targets (list[Dict]): List of target dictionaries containing 'boxes', 'labels', and 'study_label'.

        Returns:
            Training: Dict of losses (RPN losses, ROI losses, Global Classification loss).
            Inference: Tuple (detections, global_logits).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # 1. Capture Study Labels
        # We extract these before the transform potentially modifies the target structure
        if self.training:
            study_labels = torch.stack([t["study_label"] for t in targets])

        # 2. Standard RCNN Transform (Resize, Pad, Batching)
        # We need to capture original sizes for post-processing during inference
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # The transform updates 'boxes' in targets to match resized images
        images, targets = self.transform(images, targets)

        # 3. Backbone Forward
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # 4. Global Head Forward
        # We use the highest level feature map (usually '3' in FPN, stride 32)
        # for the broadest semantic context.
        global_feat_key = list(features.keys())[-1]
        global_feat = features[global_feat_key]
        global_logits = self.global_head(global_feat)

        global_loss = {}
        if self.training:
            loss_study = self.study_loss_fn(global_logits, study_labels)
            global_loss = {"loss_global_classifier": loss_study * self.lambda_global}

        # 5. RPN Forward
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 6. ROI Heads Forward
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        # 7. Output Construction
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(global_loss)
            return losses
        else:
            # Post-process detections (resize boxes back to original image size)
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            return detections, global_logits
