import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularEmbedding(nn.Module):
    """
    Processes clinical metadata to produce an embedding vector.
    """

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(Config.TABULAR_INPUT_DIM, Config.TABULAR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.TABULAR_HIDDEN_DIM, Config.ATTENTION_DIM),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class DualVisualBackbone(nn.Module):
    """
    Two independent EfficientNet-B0 backbones for Axial and Coronal views.
    Extracts global features and projects them to the attention dimension.
    """

    def __init__(self):
        super().__init__()
        # Load pretrained EfficientNet-B0
        self.backbone_ax = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            in_chans=Config.IN_CHANNELS,
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            in_chans=Config.IN_CHANNELS,
        )

        # Determine feature dimension (1280 for EfficientNet-B0)
        self.feat_dim = self.backbone_ax.num_features

        # Projection layer to match Attention Dimension
        self.proj = nn.Linear(self.feat_dim, Config.ATTENTION_DIM)

    def forward(self, img_ax, img_cor):
        # Extract features (Batch, Feat_Dim)
        f_ax = self.backbone_ax(img_ax)
        f_cor = self.backbone_cor(img_cor)

        # Project to Attention Dimension (Batch, Attn_Dim)
        k_ax = self.proj(f_ax)
        k_cor = self.proj(f_cor)

        return k_ax, k_cor


class ResidualCrossAttentionNet(nn.Module):
    """
    Architecture based on Symmetric Self-Attention with Skip Connections.
    Cite {solution_lesson_node_00018} and {solution_lesson_node_00028}.

    1. Visual Backbones -> Visual Embeddings
    2. Tabular MLP -> Tabular Embedding
    3. Symmetric Self-Attention (Visual + Tabular)
    4. Skip Connection: Concat(Fused Context, Tabular Embedding)
    5. Regression Head -> [alpha, sigma_base, sigma_growth]
    """

    def __init__(self):
        super().__init__()
        self.tabular_embedding = TabularEmbedding()
        self.visual_backbone = DualVisualBackbone()

        # Symmetric Self-Attention
        # Embed dim = Config.ATTENTION_DIM
        self.attention = nn.MultiheadAttention(
            embed_dim=Config.ATTENTION_DIM, num_heads=4, batch_first=True
        )

        # Regression Head
        # Input: Fused Context (Attn Dim) + Skip Tabular (Attn Dim)
        input_dim = Config.ATTENTION_DIM * 2

        self.head = nn.Sequential(
            nn.Linear(input_dim, Config.TABULAR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.TABULAR_HIDDEN_DIM, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(self, tabular, img_ax, img_cor, relative_week, baseline_fvc):
        """
        Args:
            tabular: (B, 7)
            img_ax: (B, 3, 224, 224)
            img_cor: (B, 3, 224, 224)
            relative_week: (B,) or (B, 1)
            baseline_fvc: (B,) or (B, 1)
        """
        # 1. Embeddings
        tab_emb = self.tabular_embedding(tabular)  # (B, 128)
        vis_ax, vis_cor = self.visual_backbone(img_ax, img_cor)  # (B, 128) each

        # 2. Symmetric Self-Attention
        # Create sequence: [Axial, Coronal, Tabular]
        # Shape: (B, 3, 128)
        sequence = torch.stack([vis_ax, vis_cor, tab_emb], dim=1)

        # Self-Attention
        # attn_output: (B, 3, 128)
        attn_output, _ = self.attention(sequence, sequence, sequence)

        # 3. Pooling / Context
        # We pool the attention output to get a single context vector.
        # Mean pooling over the sequence dimension.
        context = torch.mean(attn_output, dim=1)  # (B, 128)

        # 4. Skip Connection
        # Concatenate Context with original Tabular Embedding
        # Cite {solution_lesson_node_00018}
        fused = torch.cat([context, tab_emb], dim=1)  # (B, 256)

        # 5. Prediction
        params = self.head(fused)

        alpha = params[:, 0]
        sigma_base = params[:, 1]
        sigma_growth = params[:, 2]

        # 6. Constraints
        # Cite {solution_lesson_node_00021}
        sigma_base = F.softplus(sigma_base)
        sigma_growth = F.softplus(sigma_growth)

        # 7. Parametric Inference
        if relative_week.dim() == 1:
            relative_week = relative_week.view(-1)
        if baseline_fvc.dim() == 1:
            baseline_fvc = baseline_fvc.view(-1)

        fvc_pred = baseline_fvc + alpha * relative_week
        confidence_pred = sigma_base + sigma_growth * torch.abs(relative_week)

        return fvc_pred, confidence_pred
