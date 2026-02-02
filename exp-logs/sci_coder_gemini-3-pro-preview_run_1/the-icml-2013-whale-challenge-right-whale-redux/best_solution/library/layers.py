import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


# ==========================================
# 1. Coordinate Attention
# ==========================================
class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Block.
    Captures long-range dependencies by aggregating features along spatial directions.
    """

    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pool
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Expand
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out


class CABlockWrapper(nn.Module):
    """
    Wraps a standard ResNet BasicBlock to add Coordinate Attention.
    """

    def __init__(self, block, channels, reduction=32):
        super(CABlockWrapper, self).__init__()
        self.block = block
        self.ca = CoordinateAttention(channels, reduction)

    def forward(self, x):
        out = self.block(x)
        out = self.ca(out)
        return out


# ==========================================
# 2. Context-Gated Spectral Pooling
# ==========================================
class ContextGatedSpectralPooling(nn.Module):
    """
    Context-Gated Spectral Pooling.
    Uses the deepest layer (L4) as a semantic context to generate attention weights
    for frequency bins of shallower layers (L2, L3) before pooling.
    """

    def __init__(self, c2, f2, c3, f3, c4, reduction=16):
        super(ContextGatedSpectralPooling, self).__init__()

        # Context Processing (from L4)
        self.context_process = nn.Sequential(
            nn.Conv1d(c4, c4 // 2, kernel_size=1),
            nn.BatchNorm1d(c4 // 2),
            nn.ReLU(inplace=True),
        )
        ctx_dim = c4 // 2

        # Attention Generators
        # Generates weights for F2 frequency bins
        self.att_gen_l2 = nn.Sequential(
            nn.Conv1d(ctx_dim, f2, kernel_size=1), nn.Sigmoid()
        )
        # Generates weights for F3 frequency bins
        self.att_gen_l3 = nn.Sequential(
            nn.Conv1d(ctx_dim, f3, kernel_size=1), nn.Sigmoid()
        )

        # Fusion
        fused_dim = c2 + c3 + c4
        self.fusion = nn.Sequential(
            nn.Conv1d(fused_dim, fused_dim // 2, kernel_size=1),
            nn.BatchNorm1d(fused_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv1d(fused_dim // 2, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
        )

        # Squeeze-and-Excitation on fused features
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(256, 256 // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256 // 16, 256, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x2, x3, x4):
        """
        x2: (B, C2, F2, T)
        x3: (B, C3, F3, T)
        x4: (B, C4, F4, T)
        """
        B, C2, F2, T = x2.shape
        _, C3, F3, _ = x3.shape
        _, C4, F4, _ = x4.shape

        # 1. Generate Context from L4
        # GAP over Frequency: (B, C4, F4, T) -> (B, C4, T)
        ctx_raw = x4.mean(dim=2)
        ctx = self.context_process(ctx_raw)  # (B, ctx_dim, T)

        # 2. Generate Attention Weights
        # (B, ctx_dim, T) -> (B, F2, T)
        att_l2 = self.att_gen_l2(ctx)
        # (B, ctx_dim, T) -> (B, F3, T)
        att_l3 = self.att_gen_l3(ctx)

        # 3. Apply Gated Pooling
        # L2: (B, C2, F2, T) * (B, 1, F2, T) -> Sum over F -> (B, C2, T)
        x2_gated = x2 * att_l2.unsqueeze(1)
        x2_pool = x2_gated.sum(dim=2)

        # L3: (B, C3, F3, T) * (B, 1, F3, T) -> Sum over F -> (B, C3, T)
        x3_gated = x3 * att_l3.unsqueeze(1)
        x3_pool = x3_gated.sum(dim=2)

        # L4: Standard GAP over Freq -> (B, C4, T)
        x4_pool = ctx_raw

        # 4. Fusion
        # Concat: (B, C2+C3+C4, T)
        fused = torch.cat([x2_pool, x3_pool, x4_pool], dim=1)
        out = self.fusion(fused)  # (B, 256, T)

        # SE Block
        se_w = self.se(out)
        out = out * se_w

        return out


# ==========================================
# 3. Attention Pooling
# ==========================================
class AttentionPooling(nn.Module):
    """
    Attention Pooling for Temporal Aggregation.
    Aggregates the RNN sequence into a single embedding vector.
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
        # x: (B, T, C)
        weights = self.attention(x)  # (B, T, 1)
        out = torch.sum(x * weights, dim=1)  # (B, C)
        return out


# ==========================================
# 4. Main Model: ContextGatedResNet18
# ==========================================
class ContextGatedResNet18(nn.Module):
    def __init__(self, config=None):
        super(ContextGatedResNet18, self).__init__()

        # 1. Backbone: ResNet18
        # We load pretrained weights
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify first conv for 1-channel input (average weights)
        original_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv1

        # 2. Modify Strides for L3 and L4 (Asymmetric: Freq=2, Time=1)
        # Layer 3
        backbone.layer3[0].conv1.stride = (2, 1)
        backbone.layer3[0].downsample[0].stride = (2, 1)
        # Layer 4
        backbone.layer4[0].conv1.stride = (2, 1)
        backbone.layer4[0].downsample[0].stride = (2, 1)

        # 3. Inject Coordinate Attention
        # Wrap each BasicBlock in layers 1-4
        self.layer1 = nn.Sequential(*[CABlockWrapper(b, 64) for b in backbone.layer1])
        self.layer2 = nn.Sequential(*[CABlockWrapper(b, 128) for b in backbone.layer2])
        self.layer3 = nn.Sequential(*[CABlockWrapper(b, 256) for b in backbone.layer3])
        self.layer4 = nn.Sequential(*[CABlockWrapper(b, 512) for b in backbone.layer4])

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )

        # 4. Determine Feature Dimensions (Dummy Pass)
        # Input: (1, 1, 128, 125) -> based on Config
        # We need to know F2, F3, F4
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 125)
            x = self.stem(dummy)
            l1 = self.layer1(x)
            l2 = self.layer2(l1)  # C=128
            l3 = self.layer3(l2)  # C=256
            l4 = self.layer4(l3)  # C=512

            f2 = l2.shape[2]
            f3 = l3.shape[2]
            # f4 = l4.shape[2] # Not strictly needed for projection but good to know

        # 5. Context-Gated Spectral Pooling
        self.cg_pool = ContextGatedSpectralPooling(c2=128, f2=f2, c3=256, f3=f3, c4=512)

        # 6. RNN Head
        # Input dim is 256 (from cg_pool fusion)
        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # 7. Attention Pooling & Classifier
        self.att_pool = AttentionPooling(128 * 2)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 * 2, 1))

    def forward(self, x):
        # x: (B, 1, F, T)

        # Backbone
        x = self.stem(x)
        x = self.layer1(x)

        x2 = self.layer2(x)  # (B, 128, F2, T_sub)
        x3 = self.layer3(x2)  # (B, 256, F3, T_sub)
        x4 = self.layer4(x3)  # (B, 512, F4, T_sub)

        # Context-Gated Pooling
        # Returns (B, 256, T_sub)
        x_fused = self.cg_pool(x2, x3, x4)

        # Permute for RNN: (B, T, C)
        x_seq = x_fused.permute(0, 2, 1)

        # RNN
        self.rnn.flatten_parameters()
        x_rnn, _ = self.rnn(x_seq)  # (B, T, 256)

        # Attention Pooling
        x_emb = self.att_pool(x_rnn)  # (B, 256)

        # Classifier
        logits = self.fc(x_emb)

        return logits
