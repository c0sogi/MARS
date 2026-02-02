import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class MIPLinearDecayNet(nn.Module):
    """
    A multi-modal neural network that predicts the rate of lung function decline (slope)
    and a confidence measure (uncertainty) based on a CT scan MIP image and clinical metadata.
    """

    def __init__(self):
        super(MIPLinearDecayNet, self).__init__()

        # ==========================
        # 1. Image Backbone (ResNet18)
        # ==========================
        # Load pretrained weights if specified in Config
        weights = models.ResNet18_Weights.DEFAULT if Config.PRETRAINED else None
        self.backbone = models.resnet18(weights=weights)

        # Modify the first convolution layer to accept 1 channel (Grayscale MIP) instead of 3 (RGB)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        # Initialize the new conv1 weights by summing the original RGB weights
        # This preserves the intensity response of the filters better than random initialization
        if weights is not None:
            with torch.no_grad():
                self.backbone.conv1.weight.data = original_conv1.weight.data.sum(
                    dim=1, keepdim=True
                )

        # Remove the original fully connected classification layer
        # ResNet18 outputs a 512-dimensional feature vector after the average pooling
        self.n_img_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # ==========================
        # 2. Tabular MLP
        # ==========================
        # Processes the clinical features (Age, Sex, Smoking, Percent)
        self.n_tab_features = Config.N_TABULAR_FEATURES
        self.tab_hidden_dim = 128

        self.tabular_net = nn.Sequential(
            nn.Linear(self.n_tab_features, self.tab_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.tab_hidden_dim, self.tab_hidden_dim),
            nn.ReLU(),
        )

        # ==========================
        # 3. Fusion Head
        # ==========================
        # Concatenates Image and Tabular embeddings -> Predicts Slope and Confidence
        fusion_dim = self.n_img_features + self.tab_hidden_dim
        hidden_dim = Config.HIDDEN_DIM

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Regularization to prevent overfitting
            nn.Linear(hidden_dim, 2),  # Output: [Slope, Confidence]
        )

    def forward(self, image, tabular):
        """
        Forward pass of the network.

        Args:
            image (Tensor): Batch of MIP images. Shape (B, 1, H, W).
            tabular (Tensor): Batch of tabular features. Shape (B, N_TABULAR_FEATURES).

        Returns:
            slope (Tensor): Predicted rate of decline (alpha). Shape (B, 1).
            confidence (Tensor): Predicted uncertainty (sigma). Shape (B, 1).
        """
        # --- Image Branch ---
        # Extract features from the MIP image
        img_feat = self.backbone(image)  # Output shape: (B, 512)

        # --- Tabular Branch ---
        # Extract features from clinical metadata
        tab_feat = self.tabular_net(tabular)  # Output shape: (B, 128)

        # --- Fusion ---
        # Concatenate features along the channel dimension
        concat_feat = torch.cat((img_feat, tab_feat), dim=1)  # Output shape: (B, 640)

        # --- Prediction ---
        output = self.head(concat_feat)  # Output shape: (B, 2)

        # Split output into Slope and Confidence
        # Slope: Unbounded linear output (index 0)
        slope = output[:, 0].unsqueeze(1)

        # Confidence: Must be positive (index 1)
        # We use Softplus (smooth approximation of ReLU) to ensure positivity
        confidence = F.softplus(output[:, 1].unsqueeze(1))

        return slope, confidence
