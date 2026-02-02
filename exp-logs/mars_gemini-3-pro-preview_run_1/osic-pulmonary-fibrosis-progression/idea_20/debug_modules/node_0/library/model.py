import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ModalityAwareDualAxisNet(nn.Module):
    """
    Modality-Aware Symmetric Dual-Axis Network.

    Features:
    - Two independent EfficientNet-B0 backbones for Axial and Coronal views.
    - Tabular MLP projecting clinical features to visual dimension.
    - Learnable Modality Embeddings added to feature vectors.
    - Symmetric Multi-Head Self-Attention Fusion.
    - Prior-Anchored Head predicting trajectory parameters (alpha, sigma_base, sigma_growth).
    """

    def __init__(self, tabular_input_dim=7, embedding_dim=1280):
        super(ModalityAwareDualAxisNet, self).__init__()

        # 1. Independent Visual Backbones
        # EfficientNet-B0 pretrained on ImageNet.
        # num_classes=0 with global_pool='' (default avg) returns the pooled feature vector (1280-dim).
        self.axial_backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )
        self.coronal_backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )

        # 2. Up-Projected Tabular Embedding
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, 512), nn.ReLU(), nn.Linear(512, embedding_dim)
        )

        # 3. Modality Embeddings
        # Learnable vectors for [Axial, Coronal, Tabular] to resolve ambiguity in symmetric attention
        self.modality_embeddings = nn.Parameter(torch.randn(1, 3, embedding_dim))

        # 4. Symmetric Attention Fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=8, batch_first=True
        )

        # 5. Prior-Anchored Head
        # Input: Fused Context (1280) + Raw Tabular Skip Connection (7)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim + tabular_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3),  # Outputs: alpha (slope), sigma_base, sigma_growth
        )

    def forward(self, img_axial, img_coronal, tabular, meta):
        """
        Args:
            img_axial (torch.Tensor): (Batch, 3, 224, 224)
            img_coronal (torch.Tensor): (Batch, 3, 224, 224)
            tabular (torch.Tensor): (Batch, 7) - Normalized clinical features
            meta (torch.Tensor): (Batch, 3) - [Baseline_FVC, Baseline_Week, Current_Week]

        Returns:
            pred_fvc (torch.Tensor): Predicted FVC (Batch,)
            pred_sigma (torch.Tensor): Predicted Confidence (Batch,)
        """
        batch_size = img_axial.size(0)

        # 1. Feature Extraction
        # Backbones return (B, 1280)
        feat_ax = self.axial_backbone(img_axial)
        feat_cor = self.coronal_backbone(img_coronal)

        # Tabular projection returns (B, 1280)
        feat_tab = self.tabular_mlp(tabular)

        # 2. Modality-Aware Sequence Construction
        # Stack to (B, 3, 1280) corresponding to [Axial, Coronal, Tabular]
        seq = torch.stack([feat_ax, feat_cor, feat_tab], dim=1)

        # Add learnable embeddings (broadcasting across batch)
        seq = seq + self.modality_embeddings

        # 3. Fusion
        # Self-Attention: (B, Seq_Len, Embed_Dim)
        attn_out, _ = self.attention(seq, seq, seq)

        # Global Average Pooling over the sequence dimension -> (B, 1280)
        context = torch.mean(attn_out, dim=1)

        # 4. Prediction Head
        # Concatenate fused context with raw tabular features (Skip Connection)
        combined = torch.cat([context, tabular], dim=1)  # (B, 1287)

        # Predict parameters
        params = self.head(combined)

        # Extract individual parameters
        alpha = params[:, 0]  # Slope can be negative
        sigma_base = F.softplus(params[:, 1])  # Uncertainty must be positive
        sigma_growth = F.softplus(params[:, 2])  # Uncertainty growth must be positive

        # 5. Anchored Trajectory Logic
        # meta: [Baseline_FVC, Baseline_Week, Current_Week]
        baseline_fvc = meta[:, 0]
        baseline_week = meta[:, 1]
        current_week = meta[:, 2]

        dt = current_week - baseline_week

        # Linear Trajectory
        pred_fvc = baseline_fvc + alpha * dt

        # Uncertainty Trajectory
        pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

        return pred_fvc, pred_sigma
