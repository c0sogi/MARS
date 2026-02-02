import torch
import torch.nn as nn
import timm
from library.config import Config


class BraTSClassifier(nn.Module):
    """
    2.5D EfficientNet with Learnable Multi-Slice Projection (LMSP).

    This model takes a 'thick slab' of MRI slices (multiple slices per modality) as input.
    It uses a learnable 1x1 convolution (Adapter) to project these slices into a 3-channel
    tensor (RGB-like) which is then processed by a standard ImageNet-pretrained EfficientNet.

    The adapter is initialized to pass through the middle slice of each modality, ensuring
    the model starts with a strong 2D baseline and learns to utilize the volumetric context
    (neighboring slices) during training.
    """

    def __init__(self):
        super(BraTSClassifier, self).__init__()

        # ==========================================
        # 1. Learnable Input Adapter
        # ==========================================
        # Projects IN_CHANNELS (e.g., 9) down to 3 (RGB-like) for the backbone.
        # Kernel size 1x1 acts as a learnable linear combination of slices/modalities per pixel.
        self.adapter = nn.Conv2d(
            in_channels=Config.IN_CHANNELS, out_channels=3, kernel_size=1, bias=True
        )

        # ==========================================
        # 2. Backbone (EfficientNet-B0)
        # ==========================================
        # Load pretrained EfficientNet-B0.
        # num_classes=1 for binary classification (MGMT promoter methylation).
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
        )

        # ==========================================
        # 3. Initialization
        # ==========================================
        self.init_adapter_weights()

    def init_adapter_weights(self):
        """
        Initializes the adapter layer to act as an Identity Pass-Through for the middle slices.

        Logic:
            - Weights are initialized to 0.
            - For each modality (FLAIR, T1wCE, T2w), the weight corresponding to the
              middle slice is set to 1.0 in the corresponding output channel (R, G, or B).
            - This effectively copies the middle slices to the RGB input of the backbone
              at initialization, preserving ImageNet priors.
        """
        # Zero out all weights and biases first
        nn.init.constant_(self.adapter.weight, 0.0)
        if self.adapter.bias is not None:
            nn.init.constant_(self.adapter.bias, 0.0)

        # Calculate indices
        middle_offset = Config.SLICE_DEPTH // 2
        num_modalities = len(Config.MODALITY_COLS)

        # We map up to 3 modalities to the 3 output channels (RGB)
        # Config.MODALITY_COLS = ["flair_path", "t1wce_path", "t2w_path"]
        # Output Channel 0 (R) <- FLAIR Middle
        # Output Channel 1 (G) <- T1wCE Middle
        # Output Channel 2 (B) <- T2w Middle

        with torch.no_grad():
            for m in range(min(num_modalities, 3)):
                # Calculate the input channel index for the middle slice of modality 'm'
                # Input structure: [Mod0_S0, Mod0_S1, Mod0_S2, Mod1_S0, ...]
                input_idx = m * Config.SLICE_DEPTH + middle_offset

                # Output channel index (0, 1, or 2)
                output_idx = m

                # Set weight to 1.0 (Identity)
                self.adapter.weight[output_idx, input_idx, 0, 0] = 1.0

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, 9, H, W)
               Contains stacked slices: [FLAIR_slices, T1wCE_slices, T2w_slices]
        Returns:
            logits: Output tensor of shape (Batch, 1)
        """
        # 1. Project 9 channels -> 3 channels (Synthesize 'Super-RGB' image)
        x = self.adapter(x)

        # 2. Pass through backbone
        # timm efficientnet returns (Batch, num_classes)
        x = self.backbone(x)

        return x
