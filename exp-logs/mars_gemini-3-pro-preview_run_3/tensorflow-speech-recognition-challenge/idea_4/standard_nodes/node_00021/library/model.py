import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library import config


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer to aggregate the temporal sequence.
    Learns to weight important time steps (speech) higher than unimportant ones (silence).
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x shape: (Batch, Time, Features)

        # Calculate attention weights
        # weights shape: (Batch, Time, 1)
        weights = self.attention(x)

        # Weighted sum
        # context shape: (Batch, Features)
        context = torch.sum(x * weights, dim=1)

        return context


class MultiResResNetCRNN(nn.Module):
    """
    Multi-Resolution ResNet34-CRNN.

    Input: (B, 3, F, T) - 3-channel Multi-Resolution Log-Mel Spectrogram.
    Backbone: ResNet34 (pretrained), with modified strides in Layer 3 & 4.
    Neck: Bidirectional GRU.
    Head: Attention Pooling + Linear Classifier.
    """

    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=True):
        super(MultiResResNetCRNN, self).__init__()

        # 1. Load Pretrained Backbone
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)

        # 2. Modify Strides to preserve Temporal Resolution
        # Standard ResNet downsamples by 32x. We reduce this to 8x in the time dimension
        # by changing strides in layer3 and layer4 to 1.

        # Layer 3 modification
        resnet.layer3[0].conv1.stride = (1, 1)
        if resnet.layer3[0].downsample is not None:
            resnet.layer3[0].downsample[0].stride = (1, 1)

        # Layer 4 modification
        resnet.layer4[0].conv1.stride = (1, 1)
        if resnet.layer4[0].downsample is not None:
            resnet.layer4[0].downsample[0].stride = (1, 1)

        # 3. Extract Layers
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # ResNet34 layer4 output channels = 512
        self.cnn_out_dim = resnet.fc.in_features

        # 4. Recurrent Neck
        self.rnn = nn.GRU(
            input_size=self.cnn_out_dim,
            hidden_size=config.RNN_HIDDEN_SIZE,
            num_layers=config.RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.DROPOUT if config.RNN_LAYERS > 1 else 0,
        )

        rnn_out_dim = config.RNN_HIDDEN_SIZE * 2

        # 5. Attention Head
        self.attention_pooling = AttentionPooling(rnn_out_dim)
        self.classifier = nn.Linear(rnn_out_dim, num_classes)

    def forward(self, x):
        # Input: (Batch, 3, Freq, Time)

        # --- CNN Backbone ---
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Output: (Batch, 512, F', T')

        # --- Prepare for RNN ---
        # Average over Frequency dimension
        x = torch.mean(x, dim=2)  # (Batch, 512, T')

        # Permute to (Batch, Time, Features) for RNN
        x = x.permute(0, 2, 1)

        # --- RNN ---
        # self.rnn returns (output, h_n)
        # output shape: (Batch, Time, 2*Hidden)
        x, _ = self.rnn(x)

        # --- Attention Pooling ---
        # Aggregates time dimension
        x = self.attention_pooling(x)  # (Batch, 2*Hidden)

        # --- Classifier ---
        logits = self.classifier(x)  # (Batch, NumClasses)

        return logits
