import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import CFG


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ArcMarginProduct(nn.Module):
    """
    ArcFace head for metric learning.
    """

    def __init__(
        self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i}) ----------------

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class WhaleModel(nn.Module):
    """
    Main model class for Whale Species Prediction.
    Integrates Backbone -> GeM -> Neck -> ArcFace Head.
    """

    def __init__(self, model_name, num_classes, pretrained=True):
        super(WhaleModel, self).__init__()

        # Create Backbone
        # global_pool='' ensures we get the spatial features (B, C, H, W) for GeM
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Enable Gradient Checkpointing if requested
        if CFG.use_gradient_checkpointing:
            # Most timm models support this method
            if hasattr(self.backbone, "set_grad_checkpointing"):
                self.backbone.set_grad_checkpointing(enable=True)
            else:
                print(
                    f"Warning: {model_name} does not support set_grad_checkpointing in timm."
                )

        # Determine backbone output features
        self.in_features = self.backbone.num_features

        # Pooling Layer
        self.pooling = GeM(p=3)

        # Neck: Projection to embedding size (e.g., 2048 -> 512)
        # Using a Linear layer + BatchNorm is standard for Metric Learning necks
        self.neck = nn.Sequential(
            nn.Linear(self.in_features, CFG.embedding_size, bias=False),
            nn.BatchNorm1d(CFG.embedding_size),
        )

        # Head: ArcFace
        self.head = ArcMarginProduct(
            CFG.embedding_size, num_classes, s=CFG.arcface_s, m=CFG.arcface_m
        )

    def forward(self, images, labels=None):
        """
        Forward pass.

        Args:
            images (torch.Tensor): Input images.
            labels (torch.Tensor, optional): Target labels. Required for training with ArcFace.

        Returns:
            If labels is not None (Training): Returns ArcFace logits.
            If labels is None (Inference): Returns normalized embeddings.
        """
        # Feature extraction
        features = self.backbone(images)

        # Pooling
        features = self.pooling(features)

        # Flatten
        features = features.flatten(1)

        # Projection (Neck)
        embeddings = self.neck(features)

        if labels is not None:
            # Training: Apply ArcFace margin
            logits = self.head(embeddings, labels)
            return logits
        else:
            # Inference: Return embeddings (often normalized for cosine similarity)
            # ArcFace head normalizes internally, but for inference we usually
            # want the output of the neck, possibly normalized.
            # Normalizing here ensures consistency with the cosine metric.
            return F.normalize(embeddings)
