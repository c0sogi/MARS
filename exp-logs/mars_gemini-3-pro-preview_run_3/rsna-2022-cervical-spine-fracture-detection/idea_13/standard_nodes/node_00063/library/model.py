import torch
import torch.nn as nn
import timm
from library.config import Config


class RSNAModel(nn.Module):
    """
    ResNet18 Multi-Task MIL Network.
    Simplified architecture to enable larger batch sizes and stable training.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone
        # ResNet18 is lightweight and works well with Batch Size >= 8. Cite solution_lesson_node_00022
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0
        )

        self.backbone_dim = self.backbone.num_features

        # 2. Context Module
        # 1D Convolution to capture Z-axis continuity.
        # Using BatchNorm and ReLU as per Lesson 00027.
        self.context_conv = nn.Conv1d(
            in_channels=self.backbone_dim,
            out_channels=Config.HIDDEN_DIM,
            kernel_size=3,
            padding=1,
        )
        self.context_norm = nn.BatchNorm1d(Config.HIDDEN_DIM)
        self.context_act = nn.ReLU()

        # 3. Multi-Task Heads
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_CLASSES)

    def forward(self, global_input):
        """
        Args:
            global_input (Tensor): (Batch, Slices, 3, H, W).

        Returns:
            Tensor: (Batch, Num_Classes) - Exam-level logits.
        """
        b, s, c, h, w = global_input.shape

        # Combine Batch and Slice dimensions
        # Shape: (B*S, 3, H, W)
        x = global_input.view(b * s, c, h, w)

        # Backbone
        # Shape: (B*S, Backbone_Dim)
        x = self.backbone(x)

        # Reshape back to sequence
        # Shape: (B, S, Backbone_Dim)
        x = x.view(b, s, -1)

        # Context Module
        # Input: (B, Backbone_Dim, S)
        x = x.permute(0, 2, 1)

        # Conv1d
        x = self.context_conv(x)
        # BatchNorm1d
        x = self.context_norm(x)
        # ReLU
        x = self.context_act(x)

        # Permute back: (B, S, Hidden_Dim)
        x = x.permute(0, 2, 1)

        # Heads
        instance_logits = self.head(x)

        # Aggregation (Global Max Pooling)
        exam_logits, _ = torch.max(instance_logits, dim=1)

        return exam_logits
