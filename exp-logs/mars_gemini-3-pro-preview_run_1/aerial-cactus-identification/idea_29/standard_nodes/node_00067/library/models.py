import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------------------------------------------------------
# RepVGG Components
# -------------------------------------------------------------------------


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        deploy=False,
    ):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.non_linearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.non_linearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.non_linearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        )

    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (
            (
                self.rbr_dense.bn.weight
                / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())
            )
            .reshape(-1, 1, 1, 1)
            .detach()
        )
        t1 = (
            (
                self.rbr_1x1.bn.weight
                / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())
            )
            .reshape(-1, 1, 1, 1)
            .detach()
        )

        l2_loss_circle = (K3**2).sum() - (K3.mean(dim=[2, 3]) ** 2).sum() * K3.shape[
            2
        ] * K3.shape[3]
        return l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


class CactusRepVGG(nn.Module):
    """
    RepVGG-style backbone optimized for 32x32 input.
    Includes Auxiliary Head for file size regression.
    """

    def __init__(self, num_classes=1, width_multiplier=[1, 1, 1, 1]):
        super(CactusRepVGG, self).__init__()

        # Base width for stages
        self.in_planes = min(64, int(64 * width_multiplier[0]))

        # Stage 0: Stem (No pooling, stride 1)
        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            deploy=False,
        )

        # Stage 1
        planes1 = int(64 * width_multiplier[0])
        self.stage1 = self._make_stage(
            planes1, num_blocks=2, stride=1
        )  # Keep resolution high early on

        # Stage 2
        planes2 = int(128 * width_multiplier[1])
        self.stage2 = self._make_stage(planes2, num_blocks=2, stride=2)  # 16x16

        # Stage 3
        planes3 = int(256 * width_multiplier[2])
        self.stage3 = self._make_stage(planes3, num_blocks=2, stride=2)  # 8x8

        # Stage 4
        planes4 = int(512 * width_multiplier[3])
        self.stage4 = self._make_stage(planes4, num_blocks=2, stride=2)  # 4x4

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # Main Classification Head
        self.linear = nn.Linear(planes4, num_classes)

        # Auxiliary Regression Head (Log File Size)
        self.aux_head = nn.Sequential(
            nn.Linear(planes4, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=s,
                    padding=1,
                    deploy=False,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.gap(out)
        out = out.view(out.size(0), -1)

        logits = self.linear(out)
        aux_out = self.aux_head(out)

        return logits, aux_out


# -------------------------------------------------------------------------
# ResNet Components
# -------------------------------------------------------------------------


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CactusResNet(nn.Module):
    """
    ResNet-18 style backbone optimized for 32x32 input.
    Includes Auxiliary Head for file size regression.
    """

    def __init__(self, num_classes=1):
        super(CactusResNet, self).__init__()
        self.in_planes = 64

        # Stem: 3x3 conv, stride 1, no pooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Stages
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Main Head
        self.linear = nn.Linear(512, num_classes)

        # Auxiliary Head
        self.aux_head = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 1))

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.gap(out)
        out = out.view(out.size(0), -1)

        logits = self.linear(out)
        aux_out = self.aux_head(out)

        return logits, aux_out


# -------------------------------------------------------------------------
# Trust Router
# -------------------------------------------------------------------------


class TrustRouter(nn.Module):
    """
    Gating Network for Mixture of Experts.
    Takes trust scores (errors) as input and outputs mixing weights.
    """

    def __init__(self, num_experts, hidden_dim=16):
        super(TrustRouter, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_experts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=1),
        )

    def forward(self, trust_scores):
        """
        Args:
            trust_scores: Tensor of shape (Batch, Num_Experts) containing auxiliary errors.
        Returns:
            weights: Tensor of shape (Batch, Num_Experts) summing to 1.
        """
        return self.net(trust_scores)
