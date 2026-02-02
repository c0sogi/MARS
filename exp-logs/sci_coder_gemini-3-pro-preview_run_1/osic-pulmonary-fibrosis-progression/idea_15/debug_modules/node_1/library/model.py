import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularTokenizer(nn.Module):
    """
    Projects clinical features into high-dimensional tokens to allow
    granular interaction with visual features in the Transformer.
    """

    def __init__(self, output_dim):
        super().__init__()
        # Continuous features: Age, Percent
        # Projected via learnable linear scalers
        self.age_proj = nn.Linear(1, output_dim)
        self.pct_proj = nn.Linear(1, output_dim)

        # Categorical features: Sex (2), SmokingStatus (3)
        # Projected via embeddings
        self.sex_embed = nn.Embedding(2, output_dim)
        self.smoke_embed = nn.Embedding(3, output_dim)

    def forward(self, age, sex, smoke, percent):
        """
        Args:
            age: (B,) Normalized age
            sex: (B,) Encoded sex
            smoke: (B,) Encoded smoking status
            percent: (B,) Normalized percent
        Returns:
            list of tensors: [T_age, T_sex, T_smoke, T_pct] each (B, output_dim)
        """
        # Unsqueeze continuous vars to (B, 1) for Linear layer
        t_age = self.age_proj(age.unsqueeze(1))
        t_pct = self.pct_proj(percent.unsqueeze(1))

        t_sex = self.sex_embed(sex)
        t_smoke = self.smoke_embed(smoke)

        return [t_age, t_sex, t_smoke, t_pct]


class DualAxisNet(nn.Module):
    """
    Dual-stream visual backbone processing Axial and Coronal views independently.
    """

    def __init__(self, backbone_name, pretrained):
        super().__init__()
        # Initialize two independent backbones
        # num_classes=0 ensures we get the pooled feature vector (GAP) directly
        # For EfficientNet-B0, this is 1280 dimensions
        self.axial_backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.coronal_backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )

    def forward(self, axial_img, coronal_img):
        """
        Args:
            axial_img: (B, 3, H, W)
            coronal_img: (B, 3, H, W)
        Returns:
            v_ax, v_cor: Each (B, visual_dim)
        """
        v_ax = self.axial_backbone(axial_img)
        v_cor = self.coronal_backbone(coronal_img)
        return v_ax, v_cor


class GranularTabularNetwork(nn.Module):
    """
    Granular-Tabular Symmetric Dual-Axis Network.

    Fuses dual-axis CT visual features with granular tabular tokens using
    symmetric attention, then predicts parametric decline coefficients.
    """

    def __init__(self):
        super().__init__()

        # 1. Dual Visual Backbones
        self.dual_visual = DualAxisNet(
            backbone_name=Config.BACKBONE_NAME, pretrained=Config.PRETRAINED
        )

        # EfficientNet-B0 GAP output dimension is 1280
        self.visual_dim = Config.VISUAL_DIM

        # 2. Tabular Tokenizer
        self.tokenizer = TabularTokenizer(output_dim=self.visual_dim)

        # 3. Symmetric Fusion (Transformer)
        # Sequence: [V_ax, V_cor, T_age, T_sex, T_smoke, T_pct] -> Length 6
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.visual_dim,
            nhead=Config.NUM_ATTENTION_HEADS,
            dim_feedforward=self.visual_dim * 2,
            dropout=Config.DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.NUM_ATTENTION_LAYERS
        )

        # 4. Parametric Head
        # Inputs: Pooled Fusion Vector (1280) + Raw Priors (2: FVC, Percent)
        head_input_dim = self.visual_dim + 2

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(
                512, Config.OUTPUT_DIM
            ),  # 3 outputs: alpha, sigma_base, sigma_growth
        )

    def forward(
        self, axial, coronal, age, sex, smoke, percent, priors, time_delta, **kwargs
    ):
        """
        Args:
            axial: (B, 3, 224, 224)
            coronal: (B, 3, 224, 224)
            age: (B,) Normalized Age
            sex: (B,) Encoded Sex
            smoke: (B,) Encoded SmokingStatus
            percent: (B,) Normalized Percent
            priors: (B, 2) [Baseline_FVC, Baseline_Percent] Raw scalar values
            time_delta: (B,) Time difference from baseline in weeks

        Returns:
            fvc_pred: (B,) Predicted FVC
            confidence_pred: (B,) Predicted Confidence (Sigma)
        """
        # 1. Extract Visual Features
        v_ax, v_cor = self.dual_visual(axial, coronal)  # (B, 1280) each

        # 2. Extract Tabular Tokens
        # List of 4 tensors, each (B, 1280)
        tab_tokens = self.tokenizer(age, sex, smoke, percent)

        # 3. Construct Sequence
        # Stack visual and tabular tokens: (B, 6, 1280)
        # Order: [Axial, Coronal, Age, Sex, Smoke, Percent]
        sequence_list = [v_ax, v_cor] + tab_tokens
        sequence = torch.stack(sequence_list, dim=1)

        # 4. Symmetric Attention Fusion
        # (B, 6, 1280)
        contextualized = self.transformer(sequence)

        # 5. Global Average Pooling
        # Pool across the sequence dimension to capture holistic state
        # (B, 1280)
        pooled_features = torch.mean(contextualized, dim=1)

        # 6. Skip Connection & Head
        # Concatenate with raw scalar priors (Baseline FVC, Baseline Percent)
        # This gives the head direct access to strong priors
        combined = torch.cat([pooled_features, priors], dim=1)  # (B, 1282)

        # Predict parameters
        params = self.head(combined)  # (B, 3)

        alpha = params[:, 0]
        # Enforce positivity for confidence parameters
        sigma_base = F.softplus(params[:, 1])
        sigma_growth = F.softplus(params[:, 2])

        # 7. Parametric Inference Calculation
        # FVC = Baseline + alpha * delta
        baseline_fvc = priors[:, 0]
        fvc_pred = baseline_fvc + alpha * time_delta

        # Confidence = sigma_base + sigma_growth * |delta|
        confidence_pred = sigma_base + sigma_growth * torch.abs(time_delta)

        return fvc_pred, confidence_pred
