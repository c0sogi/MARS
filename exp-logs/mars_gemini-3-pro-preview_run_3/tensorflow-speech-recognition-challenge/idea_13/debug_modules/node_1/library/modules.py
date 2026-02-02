import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import N_CLASSES, HIDDEN_DIM, DROPOUT


# ==========================================
# 1. Selective Kernel Convolution (SKConv)
# ==========================================
class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        """
        Args:
            features (int): Number of input/output channels.
            M (int): Number of branches (paths).
            G (int): Groups (cardinality).
            r (int): Reduction ratio for the fusion FC layer.
            stride (int or tuple): Stride for the convolution.
            L (int): Minimum dimension of the fusion vector.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])

        # Branch 1: 3x3, dilation 1
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(
                    features,
                    features,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    dilation=1,
                    groups=G,
                    bias=False,
                ),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            )
        )

        # Branch 2: 3x3, dilation 2 (approx 5x5 receptive field)
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(
                    features,
                    features,
                    kernel_size=3,
                    stride=stride,
                    padding=2,
                    dilation=2,
                    groups=G,
                    bias=False,
                ),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            )
        )

        # Fusion layers
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)

        # 1. Split: Compute output of each branch
        feats = [conv(x) for conv in self.convs]
        feats = torch.stack(feats, dim=1)  # (B, M, C, H, W)

        # 2. Fuse: Sum branches
        U = torch.sum(feats, dim=1)  # (B, C, H, W)

        # Global Average Pooling
        S = U.mean(-1).mean(-1)  # (B, C)
        Z = self.fc(S)  # (B, d)
        Z = F.relu(Z)

        # 3. Select: Calculate attention weights
        weights = [fc(Z).view(batch_size, self.features, 1, 1) for fc in self.fcs]
        weights = torch.stack(weights, dim=1)  # (B, M, C, 1, 1)
        weights = self.softmax(weights)  # Normalize across M branches

        # Apply weights
        V = (weights * feats).sum(dim=1)  # (B, C, H, W)
        return V


class SKBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SKBasicBlock, self).__init__()
        # First conv is SKConv
        self.conv1 = SKConv(planes, stride=stride, M=2, r=16)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # Second conv is standard 3x3 (or could be SKConv too, but typically one is enough for lightweight)
        # To keep it robust, we use standard 3x3 here to mix features
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        # Handle dimension mismatch if inplanes != planes
        if inplanes != planes and downsample is None:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ==========================================
# 2. SK-ResNet Backbone (2D Stream)
# ==========================================
class SKResNetBackbone(nn.Module):
    def __init__(self, in_channels=3, layers=[3, 4, 6, 3]):
        super(SKResNetBackbone, self).__init__()
        self.inplanes = 64

        # Stem: Asymmetric stride to preserve Time (T) while reducing Freq (F)
        # Input: (B, 3, 64, 50) -> Output: (B, 64, 32, 50)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=(2, 1), padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # MaxPool: (B, 64, 32, 50) -> (B, 64, 16, 50)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)

        # Layer 1: Stride (1,1) -> (B, 64, 16, 50)
        self.layer1 = self._make_layer(SKBasicBlock, 64, layers[0], stride=(1, 1))

        # Layer 2: Stride (2,1) -> (B, 128, 8, 50)
        self.layer2 = self._make_layer(SKBasicBlock, 128, layers[1], stride=(2, 1))

        # Layer 3: Stride (1,1) -> (B, 256, 8, 50) (Preserve resolution as per instruction)
        self.layer3 = self._make_layer(SKBasicBlock, 256, layers[2], stride=(1, 1))

        # Layer 4: Stride (1,1) -> (B, 512, 8, 50)
        self.layer4 = self._make_layer(SKBasicBlock, 512, layers[3], stride=(1, 1))

        # Output channel dimension for fusion
        self.out_dim = 512

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 3, F, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x: (B, 512, 8, T)
        # Pool over Frequency dimension
        x = torch.mean(x, dim=2)  # (B, 512, T)

        # Permute to (B, T, C) for RNN
        x = x.permute(0, 2, 1)
        return x


# ==========================================
# 3. RawNet 1D (1D Stream)
# ==========================================
class RawNet1D(nn.Module):
    def __init__(self, in_channels=1, base_filters=64):
        super(RawNet1D, self).__init__()

        # Target: Downsample 16000 -> 50 (Factor 320)
        # Strides: 4 * 4 * 4 * 5 = 320

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),  # Keep dims
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(
                base_filters, base_filters * 2, kernel_size=8, stride=4, padding=2
            ),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(
                base_filters * 2, base_filters * 4, kernel_size=8, stride=4, padding=2
            ),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(
                base_filters * 4, base_filters * 4, kernel_size=8, stride=5, padding=2
            ),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(inplace=True),
        )

        self.out_dim = base_filters * 4

    def forward(self, x):
        # x: (B, 1, L)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x: (B, C, T) -> Permute to (B, T, C)
        x = x.permute(0, 2, 1)
        return x


# ==========================================
# 4. Multi-Head Attention Pooling
# ==========================================
class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, in_dim, num_heads=4):
        super(MultiHeadAttentionPooling, self).__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        assert (
            self.head_dim * num_heads == in_dim
        ), "in_dim must be divisible by num_heads"

        # Projections
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)

        # Final projection
        self.proj = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape

        # We want to pool over T.
        # We can use a learnable query vector, or simpler self-attention.
        # Here we implement a mechanism where we compute a single attention score vector per head over T.
        # Simplified: A = Softmax(W * tanh(V * x))

        # Let's use a standard self-attention pooling approach adapted for multi-head
        # Q is a learnable parameter representing "what to look for" (global context)
        # But standard MHSA compares x to x.
        # For pooling, we usually want to reduce T->1.

        # Implementation:
        # 1. Compute scores for each head: (B, T, Heads)
        # 2. Softmax over T
        # 3. Weighted sum

        # Using a simple 2-layer MLP for scoring
        # (B, T, C) -> (B, T, Heads)
        scores = self.query(x)  # Reuse query layer name for scoring
        scores = scores.view(B, T, self.num_heads, self.head_dim)
        scores = torch.tanh(scores)
        # Project to scalar per head
        # We need a vector to dot product with. Let's just use a Linear layer reducing to num_heads
        # Re-defining structure for clarity in this specific task:

        return self._simple_attention(x)

    def _simple_attention(self, x):
        # x: (B, T, C)
        # Attention weights: (B, T, Heads)
        # We use a linear layer to compute scores for each head directly
        # Re-initialize a specific layer for this if needed, but let's use the defined ones
        # to avoid unused parameter errors, we'll implement standard self-attention
        # followed by mean pooling, OR just weighted pooling.

        # Let's implement the specific "Attention Pooling" often used in audio:
        # Att = softmax(v^T * tanh(W * x + b))

        # Multi-head version:
        # Wx: (B, T, C)
        k = self.key(x)  # (B, T, C)
        k = torch.tanh(k)

        # Project to heads: (B, T, Heads)
        # We'll use self.query as the projection vector 'v'
        # Reshape self.query to (C, Heads) effectively
        attn_logits = self.query(
            k
        )  # (B, T, C) -- reusing layers slightly differently than Transformer
        attn_logits = attn_logits.view(
            x.size(0), x.size(1), self.num_heads, -1
        )  # (B, T, H, D_h)
        attn_logits = torch.mean(attn_logits, dim=-1)  # (B, T, H)

        attn_weights = F.softmax(attn_logits, dim=1)  # Softmax over Time

        # Weighted Sum
        # x: (B, T, C) -> (B, T, H, D_h)
        x_split = x.view(x.size(0), x.size(1), self.num_heads, -1)

        # (B, T, H, 1) * (B, T, H, D_h) -> sum over T -> (B, H, D_h)
        weighted = (attn_weights.unsqueeze(-1) * x_split).sum(dim=1)

        # Concat heads: (B, C)
        out = weighted.view(x.size(0), -1)

        return out


# ==========================================
# 5. Hybrid SK-CRNN Model
# ==========================================
class HybridSKCRNN(nn.Module):
    def __init__(self):
        super(HybridSKCRNN, self).__init__()

        # Stream 1: 2D Spectral
        self.stream2d = SKResNetBackbone(in_channels=3)
        dim2d = self.stream2d.out_dim  # 512

        # Stream 2: 1D Temporal
        self.stream1d = RawNet1D(in_channels=1, base_filters=32)
        dim1d = self.stream1d.out_dim  # 32*4 = 128

        # Fusion
        fusion_dim = dim2d + dim1d

        # BiGRU
        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT,
        )

        # Attention Pooling
        self.attn_pool = MultiHeadAttentionPooling(in_dim=HIDDEN_DIM * 2, num_heads=4)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN_DIM * 2, N_CLASSES)
        )

    def forward(self, waveforms, specs):
        """
        Args:
            waveforms (torch.Tensor): (B, T_audio)
            specs (torch.Tensor): (B, 3, F, T_spec)
        """
        # 1. Process 2D Stream
        # Out: (B, T, C2d)
        feat_2d = self.stream2d(specs)

        # 2. Process 1D Stream
        # Ensure input is (B, 1, T_audio)
        if waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(1)

        # Out: (B, T, C1d)
        feat_1d = self.stream1d(waveforms)

        # 3. Align Time Dimensions
        # Due to padding differences, T might differ slightly (e.g., 50 vs 51)
        # We truncate to the minimum length
        min_t = min(feat_2d.size(1), feat_1d.size(1))
        feat_2d = feat_2d[:, :min_t, :]
        feat_1d = feat_1d[:, :min_t, :]

        # 4. Fusion
        fused = torch.cat([feat_2d, feat_1d], dim=2)  # (B, T, C_total)

        # 5. Sequence Modeling
        gru_out, _ = self.gru(fused)  # (B, T, 2*Hidden)

        # 6. Attention Pooling
        pool_out = self.attn_pool(gru_out)  # (B, 2*Hidden)

        # 7. Classification
        logits = self.classifier(pool_out)

        return logits
