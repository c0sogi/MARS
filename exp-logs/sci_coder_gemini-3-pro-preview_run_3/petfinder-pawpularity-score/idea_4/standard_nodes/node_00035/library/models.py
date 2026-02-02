import torch
import torch.nn as nn
import timm
from library.config import Config


class AdaptiveBackbone(nn.Module):
    """
    Quad-Stream Feature Extractor (Cite solution_lesson_node_00025).

    Combines:
    1. Swin Transformer Large (Supervised)
    2. EfficientNetV2 Large (Supervised)
    3. DINOv2 Large (Self-Supervised)
    4. CLIP ViT-L (Language-Contrastive)
    """

    def __init__(self, pretrained: bool = True):
        super(AdaptiveBackbone, self).__init__()

        # 1. Swin Transformer
        self.swin = timm.create_model(
            Config.BACKBONE_SWIN,
            pretrained=pretrained,
            num_classes=0,
            img_size=Config.IMG_SIZE,
        )

        # 2. EfficientNetV2
        self.effnet = timm.create_model(
            Config.BACKBONE_EFFNET, pretrained=pretrained, num_classes=0
        )

        # 3. DINOv2
        self.dino = timm.create_model(
            Config.BACKBONE_DINO,
            pretrained=pretrained,
            num_classes=0,
            img_size=Config.IMG_SIZE,
        )

        # 4. CLIP
        self.clip = timm.create_model(
            Config.BACKBONE_CLIP,
            pretrained=pretrained,
            num_classes=0,
            img_size=Config.IMG_SIZE,
        )

        # Store dimensions for independent PCA later
        self.dims = [
            self.swin.num_features,
            self.effnet.num_features,
            self.dino.num_features,
            self.clip.num_features,
        ]
        self.embedding_dim = sum(self.dims)

    def forward(self, x: torch.Tensor, feature_extract: bool = True) -> torch.Tensor:
        # Extract features from all streams
        # Using Global Average Pooling / CLS token implicitly via num_classes=0
        # Cite solution_lesson_node_00031 (No Spatial Pyramid Pooling)
        f_swin = self.swin(x)
        f_effnet = self.effnet(x)
        f_dino = self.dino(x)
        f_clip = self.clip(x)

        # Concatenate all features
        embeddings = torch.cat([f_swin, f_effnet, f_dino, f_clip], dim=1)

        return embeddings
