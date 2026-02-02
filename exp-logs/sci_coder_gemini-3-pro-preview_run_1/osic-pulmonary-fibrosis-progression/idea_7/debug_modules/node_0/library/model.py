import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class FiLMLayer(nn.Module):
    """
    Applies Feature-wise Linear Modulation (FiLM) to the input feature map.
    Formula: output = gamma * input + beta
    """

    def forward(self, x, gamma, beta):
        """
        Args:
            x: Input feature map of shape (Batch, Channels, Height, Width)
            gamma: Scale parameter of shape (Batch, Channels)
            beta: Shift parameter of shape (Batch, Channels)
        """
        # Unsqueeze gamma and beta to match spatial dimensions (B, C, 1, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)

        return gamma * x + beta


class TabularModulator(nn.Module):
    """
    MLP that processes clinical metadata to generate FiLM parameters (gamma, beta)
    for the visual backbones.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TabularModulator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DualAxisFiLMNet(nn.Module):
    """
    FiLM-Conditioned Dual-Axis Network.

    Architecture:
    1. Independent Axial and Coronal EfficientNet-B0 backbones (feature extractors).
    2. Tabular Modulator generates scale/shift params from clinical data.
    3. FiLM layers modulate the visual feature maps based on clinical context.
    4. Global Average Pooling.
    5. Skip-connection fusion (Modulated Visual + Raw Tabular).
    6. Parametric Head predicting Alpha (Slope), Sigma_Base, and Sigma_Growth.
    """

    def __init__(self):
        super(DualAxisFiLMNet, self).__init__()

        # ==========================================
        # 1. Independent Visual Backbones
        # ==========================================
        # We use EfficientNet-B0 initialized with ImageNet weights.
        # global_pool='' ensures we get the spatial feature map (B, 1280, 7, 7)
        self.axial_backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0, global_pool=""
        )

        self.coronal_backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0, global_pool=""
        )

        self.num_channels = Config.BACKBONE_OUT_CHANNELS  # 1280 for EfficientNet-B0

        # ==========================================
        # 2. Tabular Modulator
        # ==========================================
        # We need unique Gamma and Beta for EACH channel of EACH backbone.
        # Total outputs = (Channels * 2 params) * 2 backbones
        modulator_output_dim = 4 * self.num_channels

        self.modulator = TabularModulator(
            input_dim=Config.TABULAR_INPUT_DIM,
            hidden_dim=Config.FILM_HIDDEN_DIM,
            output_dim=modulator_output_dim,
        )

        # ==========================================
        # 3. FiLM & Pooling
        # ==========================================
        self.film = FiLMLayer()
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ==========================================
        # 4. Prediction Head
        # ==========================================
        # Input to head: Axial_Vector + Coronal_Vector + Raw_Tabular_Skip
        head_input_dim = (self.num_channels * 2) + Config.TABULAR_INPUT_DIM

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),  # Outputs: Alpha, Sigma_Base, Sigma_Growth
        )

    def forward(self, axial, coronal, tabular):
        """
        Args:
            axial: Tensor (B, 3, 224, 224)
            coronal: Tensor (B, 3, 224, 224)
            tabular: Tensor (B, Tabular_Input_Dim)

        Returns:
            alpha: Predicted slope
            sigma_base: Predicted base confidence
            sigma_growth: Predicted confidence growth rate
        """
        # 1. Extract Visual Features
        # Shape: (B, 1280, 7, 7)
        f_axial = self.axial_backbone(axial)
        f_coronal = self.coronal_backbone(coronal)

        # 2. Generate Modulation Parameters
        # Shape: (B, 5120)
        mod_params = self.modulator(tabular)

        # Split parameters: Gamma/Beta for Axial, Gamma/Beta for Coronal
        C = self.num_channels
        gamma_ax = mod_params[:, 0:C]
        beta_ax = mod_params[:, C : 2 * C]
        gamma_co = mod_params[:, 2 * C : 3 * C]
        beta_co = mod_params[:, 3 * C : 4 * C]

        # 3. Apply FiLM (Deep Feature Modulation)
        f_axial_mod = self.film(f_axial, gamma_ax, beta_ax)
        f_coronal_mod = self.film(f_coronal, gamma_co, beta_co)

        # 4. Global Pooling
        # Shape: (B, 1280)
        v_axial = self.pool(f_axial_mod).flatten(1)
        v_coronal = self.pool(f_coronal_mod).flatten(1)

        # 5. Fusion with Skip Connection
        # Concatenate modulated visual vectors with the raw tabular input
        combined = torch.cat([v_axial, v_coronal, tabular], dim=1)

        # 6. Prediction
        out = self.head(combined)

        # Split outputs
        alpha = out[:, 0]

        # Apply Softplus to sigmas to enforce positivity
        sigma_base = F.softplus(out[:, 1])
        sigma_growth = F.softplus(out[:, 2])

        return alpha, sigma_base, sigma_growth
