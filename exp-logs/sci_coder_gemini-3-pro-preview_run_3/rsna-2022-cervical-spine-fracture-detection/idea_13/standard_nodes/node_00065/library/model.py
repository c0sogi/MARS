import torch
import torch.nn as nn
import timm
from library.config import Config


class RSNAModel(nn.Module):
    """
    ResNet18 Multi-Task MIL Network.
    Simplified architecture to allow larger batch sizes and stable training.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone
        # Use ResNet18 for efficiency (Cite solution_lesson_node_00032)
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0
        )
        self.backbone_dim = self.backbone.num_features

        # 2. Context Module
        # Use Conv1d with BatchNorm and ReLU (Cite solution_lesson_node_00027)
        # Input channels = Backbone Dim (512)
        self.context_conv = nn.Conv1d(
            in_channels=self.backbone_dim,
            out_channels=Config.HIDDEN_DIM,
            kernel_size=3,
            padding=1,
            bias=False,  # Bias not needed with BatchNorm
        )
        self.context_norm = nn.BatchNorm1d(Config.HIDDEN_DIM)
        self.context_act = nn.ReLU(inplace=True)

        # 3. Multi-Task Heads
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (Tensor): (Batch, Slices, 3, H, W)

        Returns:
            Tensor: (Batch, Num_Classes)
        """
        b, s, c, h, w = x.shape

        # Flatten for backbone
        x = x.view(b * s, c, h, w)

        # Extract features
        x = self.backbone(x)  # (B*S, 512)

        # Reshape to sequence (B, C, S) for Conv1d
        x = x.view(b, s, -1).permute(0, 2, 1)

        # Contextualize
        x = self.context_conv(x)
        x = self.context_norm(x)
        x = self.context_act(x)

        # Reshape to (B, S, C) for Linear
        x = x.permute(0, 2, 1)

        # Classify instances
        instance_logits = self.head(x)  # (B, S, 8)

        # Global Max Pooling (Cite solution_lesson_node_00024)
        exam_logits, _ = torch.max(instance_logits, dim=1)

        return exam_logits
