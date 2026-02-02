import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiHeadAttentionPooling(nn.Module):
    """
    Implements Multi-Head Attention Pooling.
    Instead of a single global average, this learns K distinct attention masks
    to aggregate temporal features, capturing different aspects of the signal
    (e.g., onset, nucleus, coda).
    """

    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        # Projects input features to 'num_heads' attention scores
        self.attention_scores = nn.Linear(input_dim, num_heads)
        self.softmax = nn.Softmax(dim=1)  # Softmax over the time dimension

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Time, InputDim)
        Returns:
            Tensor of shape (Batch, InputDim * NumHeads)
        """
        # 1. Compute attention scores: (Batch, Time, NumHeads)
        scores = self.attention_scores(x)

        # 2. Normalize scores to weights: (Batch, Time, NumHeads)
        weights = self.softmax(scores)

        # 3. Transpose for matrix multiplication: (Batch, NumHeads, Time)
        weights = weights.transpose(1, 2)

        # 4. Weighted Sum: (Batch, NumHeads, Time) @ (Batch, Time, InputDim)
        # Result: (Batch, NumHeads, InputDim)
        context = torch.bmm(weights, x)

        # 5. Flatten: (Batch, NumHeads * InputDim)
        output = context.view(context.size(0), -1)

        return output


class MR_SK_CRNN(nn.Module):
    """
    Multi-Resolution Selective Kernel CRNN with Multi-Head Attention.

    Architecture:
    1. Input: 3-Channel Multi-Resolution Spectrogram (Short, Medium, Long windows).
    2. Backbone: SK-ResNet34 (Pretrained).
       - Strides in Layer 3 and 4 are modified to (1,1) to preserve time resolution.
    3. Neck: Bidirectional GRU to model sequential dependencies.
    4. Head: Multi-Head Attention Pooling -> Classifier.
    """

    def __init__(self):
        super(MR_SK_CRNN, self).__init__()

        # --- 1. Backbone: SK-ResNet34 ---
        # We load the model without the classification head and global pooling
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            in_chans=Config.IN_CHANNELS,
            num_classes=0,
            global_pool="",
        )

        # --- Stride Modification ---
        # To preserve temporal resolution for the CRNN, we remove downsampling
        # in the deeper layers (Layer 3 and Layer 4).

        # Modify Layer 3
        if hasattr(self.backbone, "layer3"):
            # The first block usually handles the downsampling
            layer3_block0 = self.backbone.layer3[0]
            # Recursively find and modify all Conv2d layers with stride (2, 2)
            # This handles both standard ResNet blocks and SKNet blocks (where convs are nested)
            for module in layer3_block0.modules():
                if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                    module.stride = (1, 1)

        # Modify Layer 4
        if hasattr(self.backbone, "layer4"):
            layer4_block0 = self.backbone.layer4[0]
            for module in layer4_block0.modules():
                if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                    module.stride = (1, 1)

        # Determine output channels of the backbone
        # Typically 512 for ResNet34 at layer 4
        with torch.no_grad():
            dummy_input = torch.randn(
                2, Config.IN_CHANNELS, Config.N_MELS, Config.TIME_STEPS
            )
            dummy_out = self.backbone(dummy_input)
            backbone_out_channels = dummy_out.shape[1]

        # --- 2. Neck: BiGRU ---
        self.gru = nn.GRU(
            input_size=backbone_out_channels,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.DROPOUT > 0 else 0,
        )

        # --- 3. Head: Multi-Head Attention Pooling ---
        # The GRU is bidirectional, so output dim is Hidden * 2
        gru_out_dim = Config.HIDDEN_DIM * 2

        self.attention_pooling = MultiHeadAttentionPooling(
            input_dim=gru_out_dim, num_heads=Config.NUM_ATTENTION_HEADS
        )

        # --- 4. Classifier ---
        # Input features = GRU_Out_Dim * Num_Heads
        classifier_input_dim = gru_out_dim * Config.NUM_ATTENTION_HEADS

        self.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(classifier_input_dim, Config.NUM_CLASSES),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (Batch, 3, F, T)
        """
        # 1. Backbone Feature Extraction
        # Shape: (Batch, C, F_out, T_out)
        x = self.backbone(x)

        # 2. Frequency Pooling
        # We average over the frequency dimension to create a sequence of vectors
        # Shape: (Batch, C, T_out)
        x = torch.mean(x, dim=2)

        # 3. Prepare for RNN
        # Permute to (Batch, Time, Channels)
        x = x.permute(0, 2, 1)

        # 4. Recurrent Processing (BiGRU)
        # Shape: (Batch, Time, Hidden*2)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)

        # 5. Multi-Head Attention Pooling
        # Shape: (Batch, Hidden*2 * NumHeads)
        x = self.attention_pooling(x)

        # 6. Classification
        # Shape: (Batch, NumClasses)
        logits = self.classifier(x)

        return logits
