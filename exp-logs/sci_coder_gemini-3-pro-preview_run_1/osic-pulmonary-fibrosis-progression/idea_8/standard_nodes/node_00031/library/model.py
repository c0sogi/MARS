import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class AttentionFusedDualAxisNet(nn.Module):
    """
    Architecture utilizing Symmetric Self-Attention and Skip Connections.
    Cite solution_lesson_node_00028 (Symmetric vs Asymmetric)
    Cite solution_lesson_node_00018 (Skip Connections)
    """

    def __init__(self):
        super().__init__()

        # 1. Tabular Encoder
        self.tab_mlp = nn.Sequential(
            nn.Linear(Config.TABULAR_INPUT_DIM, Config.TABULAR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.TABULAR_HIDDEN_DIM, Config.TABULAR_HIDDEN_DIM),
            nn.LayerNorm(Config.TABULAR_HIDDEN_DIM),
        )

        # 2. Visual Backbones (Independent)
        # Cite solution_lesson_node_00014 (Avoid Weight Sharing)
        self.backbone_ax = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            num_classes=0,
            in_chans=Config.IN_CHANNELS,
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            num_classes=0,
            in_chans=Config.IN_CHANNELS,
        )

        # Projection to match tabular dimension
        self.vis_proj = nn.Linear(
            self.backbone_ax.num_features, Config.TABULAR_HIDDEN_DIM
        )

        # 3. Symmetric Self-Attention Fusion
        # Cite solution_lesson_node_00017 (Dynamic Fusion)
        self.attn = nn.MultiheadAttention(
            embed_dim=Config.TABULAR_HIDDEN_DIM, num_heads=4, batch_first=True
        )

        # 4. Regression Head with Skip Connection
        # Input = Fused Context (128) + Tabular Skip (128) = 256
        self.head = nn.Sequential(
            nn.Linear(Config.TABULAR_HIDDEN_DIM * 2, Config.TABULAR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.TABULAR_HIDDEN_DIM, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(self, tabular, img_ax, img_cor, relative_week, baseline_fvc):
        # 1. Encode Features
        tab_emb = self.tab_mlp(tabular)  # (B, 128)

        f_ax = self.backbone_ax(img_ax)  # (B, 1280)
        f_cor = self.backbone_cor(img_cor)  # (B, 1280)

        p_ax = self.vis_proj(f_ax)  # (B, 128)
        p_cor = self.vis_proj(f_cor)  # (B, 128)

        # 2. Sequence Construction [Axial, Coronal, Tabular]
        # (B, 3, 128)
        seq = torch.stack([p_ax, p_cor, tab_emb], dim=1)

        # 3. Self-Attention
        attn_out, _ = self.attn(seq, seq, seq)

        # 4. Global Context (Average Pooling)
        context = torch.mean(attn_out, dim=1)  # (B, 128)

        # 5. Skip Connection
        # Cite solution_lesson_node_00018
        combined = torch.cat([context, tab_emb], dim=1)  # (B, 256)

        # 6. Prediction
        params = self.head(combined)

        alpha = params[:, 0]
        # Cite solution_lesson_node_00021 (Architectural Constraints)
        sigma_base = F.softplus(params[:, 1])
        sigma_growth = F.softplus(params[:, 2])

        # 7. Parametric Inference
        if relative_week.dim() == 1:
            relative_week = relative_week.view(-1)
        if baseline_fvc.dim() == 1:
            baseline_fvc = baseline_fvc.view(-1)

        fvc_pred = baseline_fvc + alpha * relative_week
        confidence_pred = sigma_base + sigma_growth * torch.abs(relative_week)

        return fvc_pred, confidence_pred
