import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer that computes a weighted sum of the input sequence.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, Config.ATTENTION_DIM),
            nn.Tanh(),
            nn.Linear(Config.ATTENTION_DIM, 1),
        )

    def forward(self, x):
        # x shape: [Batch, Time, Input_Dim]

        # Compute attention scores
        # [Batch, Time, 1]
        attn_scores = self.attention(x)

        # Normalize scores to weights
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Compute weighted sum
        # [Batch, Time, Input_Dim] * [Batch, Time, 1] -> [Batch, Time, Input_Dim]
        weighted_x = x * attn_weights

        # Sum over time dimension
        # [Batch, Input_Dim]
        pooled = torch.sum(weighted_x, dim=1)

        return pooled


class TimePreservingEfficientNet(nn.Module):
    """
    EfficientNet backbone with modified strides to preserve temporal resolution,
    followed by BiGRU and Attention Pooling.
    Cite solution_lesson_node_00025: Prioritize Objective Alignment and Signal Resolution over Backbone Complexity.
    Cite solution_lesson_node_00006: Treat Naming Mismatches as Signals of Incomplete Refactoring.
    """

    def __init__(self):
        super(TimePreservingEfficientNet, self).__init__()

        # 1. Load Backbone
        # We use in_chans=1 to handle the mono spectrogram input directly.
        # This avoids the need for manual 'conv1' patching.
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",
            in_chans=1,
        )

        # 2. Modify Strides for Time Preservation
        # EfficientNet uses 'blocks' (nn.Sequential). We iterate through them and
        # modify strides to (2, 1) where they are (2, 2) to preserve time resolution.
        # This is a heuristic to reduce downsampling in the time dimension.
        if hasattr(self.backbone, "blocks"):
            for component in self.backbone.blocks:
                for m in component.modules():
                    if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                        m.stride = (2, 1)

        # Fallback for other architectures (e.g. ResNet) if switched back in config
        elif hasattr(self.backbone, "layer2"):
            for layer_name in ["layer2", "layer3", "layer4"]:
                layer = getattr(self.backbone, layer_name, None)
                if layer:
                    for m in layer.modules():
                        if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                            m.stride = (2, 1)

        # 3. Determine Feature Dimension
        # Run a dummy forward pass to dynamically determine the feature dimension
        # Input size: [1, 1, Freq, Time]. With Hop 20, 2s -> ~200 frames.
        dummy_input = torch.zeros(1, 1, Config.N_MELS, 200)
        with torch.no_grad():
            features = self.backbone.forward_features(dummy_input)
            # Features shape: [Batch, Channels, Freq_down, Time_down]
            self.feature_dim = features.shape[1]

        # 5. Temporal Modeling (Bi-directional GRU)
        self.rnn = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
        )

        rnn_out_dim = (
            Config.RNN_HIDDEN_SIZE * 2
            if Config.BIDIRECTIONAL
            else Config.RNN_HIDDEN_SIZE
        )

        # 6. Aggregation (Attention Pooling)
        self.attn_pool = AttentionPooling(rnn_out_dim)

        # 7. Classifier
        self.fc = nn.Linear(rnn_out_dim, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrogram [Batch, 1, Freq, Time]
        Returns:
            torch.Tensor: Logits [Batch, 1]
        """
        # Backbone Feature Extraction
        # Returns [Batch, Channels, Freq_down, Time_down]
        x = self.backbone.forward_features(x)

        # Frequency Pooling
        # Average over the frequency dimension to get a sequence of feature vectors
        # [Batch, Channels, Time_down]
        x = torch.mean(x, dim=2)

        # Permute for RNN
        # [Batch, Time_down, Channels]
        x = x.permute(0, 2, 1)

        # RNN Processing
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # [Batch, Time_down, RNN_Dim]

        # Attention Pooling
        # [Batch, RNN_Dim]
        x = self.attn_pool(x)

        # Classification
        # [Batch, 1]
        logits = self.fc(x)

        return logits
