import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from library.config import Config
from library.backbone import build_backbone
from library.dino_components import DINOTransformer, SetCriterion, HungarianMatcher
from library.diagnosis_module import RoITransformerDiagnosis


class MultiTaskDINO(nn.Module):
    """
    End-to-end Multi-Task DINO architecture for Chest Radiograph Analysis.
    Combines:
    1. Swin Transformer Backbone (via library.backbone)
    2. DINO Detector (via library.dino_components)
    3. RoI-Transformer Diagnosis Module (via library.diagnosis_module)
    """

    def __init__(self, config: type = Config):
        super().__init__()
        # 1. Build Backbone
        # Returns features, masks, and position embeddings
        self.backbone = build_backbone(config)

        # 2. Build DINO Transformer (Detector)
        # Note: Config.NUM_CLASSES_DETECTION is usually 1 ('opacity')
        self.transformer = DINOTransformer(
            num_classes=config.NUM_CLASSES_DETECTION,
            hidden_dim=256,
            num_queries=config.NUM_QUERIES,
            nheads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=True,
        )

        # 3. Build Diagnosis Module (Study Classifier)
        # Uses features from backbone and boxes from DINO
        self.diagnosis_module = RoITransformerDiagnosis(config=config)

    def forward(
        self, samples: torch.Tensor, targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            samples: Batch of images (B, 3, H, W)
            targets: List of target dictionaries (required for DINO Denoising training)

        Returns:
            Dict containing:
                - 'pred_logits': (B, NQ, num_classes)
                - 'pred_boxes': (B, NQ, 4)
                - 'aux_outputs': List of dicts for intermediate layers
                - 'study_logits': (B, 4)
                - 'dn_meta': Denoising metadata
        """
        # 1. Backbone Forward
        # features: List of (B, C, H, W)
        # masks: List of (B, H, W)
        # pos: List of (B, C, H, W)
        features, masks, pos = self.backbone(samples)

        # 2. DINO Detector Forward
        # out_dino is a dict with pred_logits, pred_boxes, aux_outputs, dn_meta
        out_dino = self.transformer(features, masks, pos, targets)

        # 3. Diagnosis Module Forward
        # We use the final layer predictions for the diagnosis module
        # features: The raw feature maps from backbone
        # pred_boxes: (B, NQ, 4) normalized
        # pred_logits: (B, NQ, num_classes)
        study_logits = self.diagnosis_module(
            features=features,
            pred_boxes=out_dino["pred_boxes"],
            pred_logits=out_dino["pred_logits"],
        )

        # 4. Combine Outputs
        out_dino["study_logits"] = study_logits

        return out_dino


class MultiTaskCriterion(nn.Module):
    """
    Composite loss function for Object Detection and Study Classification.
    Wraps SetCriterion for detection and adds CrossEntropy for study.
    """

    def __init__(self, detection_criterion: SetCriterion, study_loss_coef: float = 1.0):
        super().__init__()
        self.detection_criterion = detection_criterion
        self.study_loss_coef = study_loss_coef
        self.study_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Dict from model forward
            targets: List of target dicts. Must contain 'study_label'.
        """
        # 1. Detection Losses (Box, Class, GIoU, Aux)
        loss_dict = self.detection_criterion(outputs, targets)

        # 2. Study Loss
        # outputs['study_logits']: (B, 4)
        # targets[i]['study_label']: scalar tensor
        if "study_logits" in outputs:
            study_logits = outputs["study_logits"]
            study_targets = torch.stack([t["study_label"] for t in targets])

            # Check if we are in test mode (dummy labels)
            # If all labels are -1, we skip loss (though this shouldn't happen in training)
            if (study_targets == -1).all():
                loss_study = torch.tensor(0.0, device=study_logits.device)
            else:
                loss_study = self.study_loss_fn(study_logits, study_targets)

            loss_dict["loss_study"] = loss_study * self.study_loss_coef

        return loss_dict


def build_model(config: type = Config) -> Tuple[nn.Module, nn.Module]:
    """
    Factory function to build the model and criterion.
    """
    # 1. Model
    model = MultiTaskDINO(config)

    # 2. Matcher for DINO
    matcher = HungarianMatcher(
        cost_class=config.LOSS_COEF_CLASS,
        cost_bbox=config.LOSS_COEF_BOX,
        cost_giou=config.LOSS_COEF_GIOU,
    )

    # 3. Detection Criterion
    weight_dict = {
        "loss_ce": config.LOSS_COEF_CLASS,
        "loss_bbox": config.LOSS_COEF_BOX,
        "loss_giou": config.LOSS_COEF_GIOU,
    }

    # Add weights for aux outputs
    # DINO usually has 6 decoder layers, so 5 aux outputs
    # We apply the same weights to all layers
    aux_weight_dict = {}
    for i in range(5):  # Assuming num_decoder_layers=6 -> 5 aux
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes"]

    detection_criterion = SetCriterion(
        num_classes=config.NUM_CLASSES_DETECTION,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,  # Relative classification weight of the no-object class
        losses=losses,
    )

    # 4. Multi-Task Criterion
    criterion = MultiTaskCriterion(
        detection_criterion=detection_criterion,
        study_loss_coef=config.LOSS_COEF_STUDY,
    )

    return model, criterion
