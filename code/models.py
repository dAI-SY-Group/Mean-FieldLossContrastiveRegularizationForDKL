# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24, 2026

@author: Matthias Schmal
@tui: Data Intensive Systems and Visualization Group (dAI.SY)


"""

import torch
from layer import SpectralBatchNorm2d, spectral_norm

# from models.tmp_spectral_norm_conv import spectral_norm_conv


# --------------------------------------------------------------------------- #
# Base Classes
# --------------------------------------------------------------------------- #


def reset_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        m.apply(initializer)
    if isinstance(m, (torch.nn.Linear, SpectralBatchNorm2d)):
        m.reset_parameters()
    for l in m.children():
        reset_weights(l)


def initializer(m, zero_init_residual=False):
    # Used in the pytorch implementation
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

    if zero_init_residual:
        if isinstance(m, Bottleneck) and m.bn3.weight is not None:
            torch.nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        elif isinstance(m, Block) and m.bn2.weight is not None:
            torch.nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


# Conv layers get the first sp_norm_p parameter
def conv1x1(
    in_planes, out_planes, stride=1, padding=0, bias=True, spectral_norm_param=0.0
):
    if spectral_norm_param:
        return spectral_norm(
            torch.nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            norm_multiplier=spectral_norm_param,
        )
    else:
        return torch.nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )


def conv3x3(
    in_planes, out_planes, stride=1, padding=0, bias=True, spectral_norm_param=0.0
):
    if spectral_norm_param:
        return spectral_norm(
            torch.nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            norm_multiplier=spectral_norm_param,
        )
    else:
        return torch.nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=bias,
        )


def conv5x5(
    in_planes, out_planes, stride=1, padding=0, bias=True, spectral_norm_param=0.0
):
    if spectral_norm_param:
        return spectral_norm(
            torch.nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            norm_multiplier=spectral_norm_param,
        )
    else:
        return torch.nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=5,
            stride=stride,
            padding=padding,
            bias=bias,
        )


def conv7x7(
    in_planes, out_planes, stride=1, padding=0, bias=True, spectral_norm_param=0.0
):
    if spectral_norm_param:
        return spectral_norm(
            torch.nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            norm_multiplier=spectral_norm_param,
        )
    else:
        return torch.nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=7,
            stride=stride,
            padding=padding,
            bias=bias,
        )


def linear(in_feature, out_features, bias=True, spectral_norm_param=0.0):
    if spectral_norm_param:
        return spectral_norm(
            torch.nn.Linear(in_feature, out_features, bias=bias),
            norm_multiplier=spectral_norm_param,
        )
    else:
        return torch.nn.Linear(in_feature, out_features, bias=bias)


# Normalization layers get the second sp_norm_p parameter
def normalization_with_activation(in_planes, track_running_stats=False, sp_norm_p=0.0):
    return torch.nn.Sequential(
        normalization(in_planes, track_running_stats, sp_norm_p), torch.nn.ReLU()
    )


def normalization(in_planes, track_running_stats=False, sp_norm_p=0.0):
    if sp_norm_p:
        return SpectralBatchNorm2d(
            in_planes, sp_norm_p, track_running_stats=track_running_stats
        )
    else:
        return torch.nn.BatchNorm2d(in_planes, track_running_stats=track_running_stats)


class RepeatModule2d(torch.nn.Module):
    def __init__(self, n_repeats=1):
        super(RepeatModule2d, self).__init__()
        self.n_repeats = n_repeats

    def forward(self, x):
        return x.repeat(1, self.n_repeats, 1, 1)


def get_dimension_adapter(in_planes, out_planes, stride):
    # if stride use maxpooling
    transform = torch.nn.Sequential()
    if stride > 1:
        transform.append(
            torch.nn.MaxPool2d(kernel_size=stride, stride=stride, padding=0)
        )
    if out_planes // in_planes == 2:
        transform.append(RepeatModule2d(n_repeats=2))
        return transform
    if out_planes // in_planes == 4:
        transform.append(RepeatModule2d(n_repeats=4))
        return transform
    elif out_planes // in_planes == 10:
        transform.append(RepeatModule2d(n_repeats=10))
        return transform
    elif out_planes == in_planes:
        return transform
    else:
        raise NotImplementedError(
            f"Unsupported dimension adapter for in_planes={in_planes}, out_planes={out_planes}, "
            f"stride={stride}. Only 2x and identity adapters are supported."
        )


def get_shortcut(
    in_planes,
    out_planes,
    stride,
    track_running_stats,
    shortcut_with_bn,
    bias,
    sp_norm_p,
):
    if shortcut_with_bn == 1:
        shortcut = torch.nn.Sequential(
            conv1x1(
                in_planes,
                out_planes,
                stride=stride,
                padding=0,
                bias=bias,
                spectral_norm_param=sp_norm_p[0],
            ),
            normalization(out_planes, track_running_stats, sp_norm_p[1]),
        )
        return shortcut
    elif shortcut_with_bn == 0:
        return conv1x1(
            in_planes,
            out_planes,
            stride=stride,
            padding=0,
            bias=bias,
            spectral_norm_param=sp_norm_p[0],
        )
    elif shortcut_with_bn == -1:
        return get_dimension_adapter(in_planes, out_planes, stride)
    else:
        raise NotImplementedError("shortcut_with_bn must be 0, 1, or -1. ")


# Blocks and Modules handle sp_norm_p as a tuple
class Block(torch.nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        preact=False,
        shortcut_with_bn: int = 0,
        track_running_stats: bool = False,
        bias: bool = False,
        sp_norm_p: (float, float) = (0.0, 0.0),
        dropout_rate: float = 0.0,
    ):
        super(Block, self).__init__()
        self.preact = preact
        self.conv1 = conv3x3(
            in_planes,
            planes,
            stride=stride,
            padding=1,
            bias=bias,
            spectral_norm_param=sp_norm_p[0],
        )
        self.bn1act = normalization_with_activation(
            planes, track_running_stats, sp_norm_p[1]
        )
        self.conv2 = conv3x3(
            planes,
            planes,
            stride=1,
            padding=1,
            bias=bias,
            spectral_norm_param=sp_norm_p[0],
        )
        bn2dim = in_planes if preact else planes
        self.bn2 = normalization(bn2dim, track_running_stats, sp_norm_p[1])

        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = get_shortcut(
                in_planes,
                self.expansion * planes,
                stride,
                track_running_stats,
                shortcut_with_bn,
                bias,
                sp_norm_p,
            )

        self.final_act = torch.nn.ReLU()

    def forward(self, x):
        out = x
        if self.preact:
            out = self.final_act(self.bn2(out))
        out = self.bn1act(self.conv1(out))
        if self.dropout_rate > 0.0:
            out = self.dropout(out)
        out = self.conv2(out)
        if not self.preact:
            out = self.bn2(out)
        out = out + self.shortcut(x)
        if not self.preact:
            out = self.final_act(out)
        return out


class Group(torch.nn.Module):
    """
    This class simplifies repetitive module patterns and improves the overview
    via torchinfo.summary()

        block_args = {'in_planes': , 'planes': , 'stride': ,
        'preact': , 'shortcut_with_bn': , 'track_running_stats': }
    """

    def __init__(self, block_args=None, n_blocks: int = 1):
        super().__init__()
        if block_args is None:
            raise AttributeError
        self.block_args = block_args
        self.n_blocks = n_blocks
        self.base_block = Block(**block_args)
        block_args["stride"] = 1
        block_args["in_planes"] = block_args["planes"]
        self.blocks = torch.nn.ModuleList(
            [Block(**block_args) for _ in range(n_blocks - 1)]
        )

    def forward(self, x):
        x = self.base_block(x)
        for block in self.blocks:
            x = block(x)
        return x


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        preact=False,
        shortcut_with_bn: int = 0,
        track_running_stats: bool = False,
        bias: bool = False,
        sp_norm_p: (float, float) = (0.0, 0.0),
    ):
        super(Bottleneck, self).__init__()
        self.preact = preact
        self.conv1 = conv1x1(
            in_planes, planes, bias=bias, spectral_norm_param=sp_norm_p[0]
        )
        self.bn1act = normalization_with_activation(
            planes, track_running_stats, sp_norm_p[1]
        )
        self.conv2 = conv3x3(
            planes,
            planes,
            stride=stride,
            padding=1,
            bias=bias,
            spectral_norm_param=sp_norm_p[0],
        )
        self.bn2act = normalization_with_activation(
            planes, track_running_stats, sp_norm_p[1]
        )
        self.conv3 = conv1x1(
            planes, self.expansion * planes, bias=bias, spectral_norm_param=sp_norm_p[0]
        )
        bn3dim = in_planes if preact else self.expansion * planes
        self.bn3 = normalization(bn3dim, track_running_stats, sp_norm_p[1])

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = get_shortcut(
                in_planes,
                self.expansion * planes,
                stride,
                track_running_stats,
                shortcut_with_bn,
                bias,
                sp_norm_p,
            )
        self.final_act = torch.nn.ReLU()

    def forward(self, x):
        out = x
        if self.preact:
            out = self.final_act(self.bn3(out))
        out = self.bn1act(self.conv1(out))
        out = self.bn2act(self.conv2(out))
        out = self.conv3(out)
        if not self.preact:
            out = self.bn3(out)
        out = out + self.shortcut(x)
        if not self.preact:
            out = self.final_act(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        with_classifier=True,
        preact=False,
        input_ds="tiny_imagenet",
        shortcut_with_bn: int = 0,
        track_running_stats: bool = False,
        bias: bool = False,
        sp_norm_p: (float, float) = (0.0, 0.0),
        extra_group_jump=False,
        final_dropout_rate: float = 0.0,
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.with_classifier = with_classifier
        self.preact = preact
        self.track_running_stats = track_running_stats

        self.maxpool = None
        if input_ds == "tiny_imagenet":
            self.conv1 = conv5x5(
                3, 64, stride=2, padding=3, bias=bias, spectral_norm_param=sp_norm_p[0]
            )
        elif input_ds == "cifar":
            self.conv1 = conv3x3(
                3, 64, stride=1, padding=1, bias=bias, spectral_norm_param=sp_norm_p[0]
            )
        elif input_ds == "imagenet":
            self.conv1 = conv7x7(
                3, 64, stride=2, padding=3, bias=bias, spectral_norm_param=sp_norm_p[0]
            )
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise NotImplementedError

        bn1dim = 512 if preact else 64
        self.bn1act = normalization_with_activation(
            bn1dim, track_running_stats, sp_norm_p[1]
        )
        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1,
            preact=preact,
            shortcut_with_bn=shortcut_with_bn,
            track_running_stats=track_running_stats,
            bias=bias,
            sp_norm_p=sp_norm_p,
        )
        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2,
            preact=preact,
            shortcut_with_bn=shortcut_with_bn,
            track_running_stats=track_running_stats,
            bias=bias,
            sp_norm_p=sp_norm_p,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2,
            preact=preact,
            shortcut_with_bn=shortcut_with_bn,
            track_running_stats=track_running_stats,
            bias=bias,
            sp_norm_p=sp_norm_p,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2,
            preact=preact,
            shortcut_with_bn=shortcut_with_bn,
            track_running_stats=track_running_stats,
            bias=bias,
            sp_norm_p=sp_norm_p,
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.final_dropout = (
            torch.nn.Dropout(final_dropout_rate)
            if final_dropout_rate > 0.0
            else torch.nn.Identity()
        )
        if with_classifier:
            self.classifier = linear(
                512 * block.expansion, num_classes, spectral_norm_param=sp_norm_p[0]
            )

        self.extra_group_jump = extra_group_jump
        if extra_group_jump:
            self.extra_shortcut1 = get_dimension_adapter(
                64, 64 * block.expansion, stride=1
            )
            self.extra_shortcut2 = get_dimension_adapter(
                64 * block.expansion, 128 * block.expansion, stride=2
            )
            self.extra_shortcut3 = get_dimension_adapter(
                128 * block.expansion, 256 * block.expansion, stride=2
            )
            self.extra_shortcut4 = get_dimension_adapter(
                256 * block.expansion, 512 * block.expansion, stride=2
            )

        self.apply(initializer)

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride,
        preact,
        shortcut_with_bn,
        track_running_stats,
        bias,
        sp_norm_p,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    preact,
                    shortcut_with_bn,
                    track_running_stats,
                    bias,
                    sp_norm_p,
                )
            )
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if not self.preact:
            out = self.bn1act(out)
        if self.maxpool is not None:
            out = self.maxpool(out)

        if self.extra_group_jump:
            out = self.layer1(out) + self.extra_shortcut1(out)
            out = self.layer2(out) + self.extra_shortcut2(out)
            out = self.layer3(out) + self.extra_shortcut3(out)
            out = self.layer4(out) + self.extra_shortcut4(out)
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

        if self.preact:
            out = self.bn1act(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.final_dropout(out)
        if self.with_classifier:
            out = self.classifier(out)
        return out


class WideResNet(torch.nn.Module):
    """Builds Wide ResNet.

    Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
    number of filters. Using three groups of residual blocks, the network maps
    spatial features of size 32x32 -> 16x16 -> 8x8.

    Also compare
    https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch

    Args:
      depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
        He et al. (2015)'s notation which uses the maximum depth of the network
        counting non-conv layers like dense.
      width: Integer to multiply the number of typical filters by. "k"
        in WRN-n-k.
      num_classes: Number of output classes.

    """

    def __init__(
        self,
        depth,
        width,
        num_classes,
        shortcut_with_bn=0,
        track_running_stats: bool = False,
        preact: bool = False,
        with_classifier: bool = True,
        input_ds: str = "cifar",
        bias: bool = False,
        sp_norm_p: (float, float) = (0.0, 0.0),
        dropout_rate: float = 0.0,
        extra_group_jump: bool = False,
        final_dropout_rate: float = 0.0,
    ):
        super().__init__()
        extra_group_jump = bool(extra_group_jump)
        shortcut_with_bn = int(shortcut_with_bn)
        assert (depth - 4) % 6 == 0, "depth should be 6n+4 (e.g., 16, 22, 28, 40)"
        n_blocks = (depth - 4) // 6
        self.with_classifier = with_classifier
        self.preact = preact
        self.track_running_stats = track_running_stats

        if input_ds == "tiny_imagenet":
            self.conv0 = conv5x5(
                3, 16, stride=2, padding=2, bias=bias, spectral_norm_param=sp_norm_p[0]
            )
        elif input_ds == "cifar":
            self.conv0 = conv3x3(
                3, 16, stride=1, padding=1, bias=bias, spectral_norm_param=sp_norm_p[0]
            )
        elif input_ds == "mnist":
            self.conv0 = conv3x3(
                1, 16, stride=1, padding=1, bias=bias, spectral_norm_param=sp_norm_p[0]
            )
        else:
            raise NotImplementedError

        bn0dim = 64 * width if preact else 16
        self.bn0act = normalization_with_activation(
            bn0dim, track_running_stats, sp_norm_p[1]
        )
        self.group1 = Group(
            block_args={
                "in_planes": 16,
                "planes": 16 * width,
                "stride": 1,
                "shortcut_with_bn": shortcut_with_bn,
                "preact": preact,
                "bias": bias,
                "track_running_stats": track_running_stats,
                "sp_norm_p": sp_norm_p,
                "dropout_rate": dropout_rate,
            },
            n_blocks=n_blocks,
        )
        self.group2 = Group(
            block_args={
                "in_planes": 16 * width,
                "planes": 32 * width,
                "stride": 2,
                "shortcut_with_bn": shortcut_with_bn,
                "preact": preact,
                "bias": bias,
                "track_running_stats": track_running_stats,
                "sp_norm_p": sp_norm_p,
                "dropout_rate": dropout_rate,
            },
            n_blocks=n_blocks,
        )
        self.group3 = Group(
            block_args={
                "in_planes": 32 * width,
                "planes": 64 * width,
                "stride": 2,
                "shortcut_with_bn": shortcut_with_bn,
                "preact": preact,
                "bias": bias,
                "track_running_stats": track_running_stats,
                "sp_norm_p": sp_norm_p,
                "dropout_rate": dropout_rate,
            },
            n_blocks=n_blocks,
        )

        self.extra_group_jump = extra_group_jump
        if extra_group_jump:
            self.extra_shortcut1 = get_dimension_adapter(16, 16 * width, stride=1)
            self.extra_shortcut2 = get_dimension_adapter(
                16 * width, 32 * width, stride=2
            )
            self.extra_shortcut3 = get_dimension_adapter(
                32 * width, 64 * width, stride=2
            )

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.final_dropout = (
            torch.nn.Dropout(final_dropout_rate)
            if final_dropout_rate > 0.0
            else torch.nn.Identity()
        )
        if with_classifier:
            self.classifier = linear(
                64 * width, num_classes, spectral_norm_param=sp_norm_p[0]
            )

        self.apply(initializer)

    def forward(self, x):
        out = self.conv0(x)
        if not self.preact:
            out = self.bn0act(out)
        if self.extra_group_jump:
            out = self.group1(out) + self.extra_shortcut1(out)
            out = self.group2(out) + self.extra_shortcut2(out)
            out = self.group3(out) + self.extra_shortcut3(out)
        else:
            out = self.group1(out)
            out = self.group2(out)
            out = self.group3(out)
        if self.preact:
            out = self.bn0act(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.final_dropout(out)
        if self.with_classifier:
            out = self.classifier(out)
        return out


# --------------------------------------------------------------------------- #
# ResNets ready to serve
# --------------------------------------------------------------------------- #


def SmallResNet18(
    num_classes=10,
    shortcut_with_bn=1,
    track_running_stats=False,
    preact=False,
    input_ds="cifar",
    with_classifier=True,
    bias=False,
    sp_norm_p=(0.0, 0.0),
    extra_group_jump=False,
    final_dropout_rate: float = 0.0,
):
    return WideResNet(
        10,
        2,
        num_classes,
        shortcut_with_bn,
        track_running_stats,
        preact=preact,
        input_ds=input_ds,
        with_classifier=with_classifier,
        bias=bias,
        sp_norm_p=sp_norm_p,
        extra_group_jump=extra_group_jump,
        final_dropout_rate=final_dropout_rate,
    )


def ResNet18(
    num_classes=10,
    shortcut_with_bn=1,
    track_running_stats=False,
    preact=False,
    input_ds="cifar",
    with_classifier=True,
    bias=False,
    sp_norm_p=(0.0, 0.0),
    extra_group_jump=False,
    final_dropout_rate: float = 0.0,
):
    return ResNet(
        Block,
        [2, 2, 2, 2],
        num_classes,
        with_classifier=with_classifier,
        preact=preact,
        input_ds=input_ds,
        shortcut_with_bn=shortcut_with_bn,
        track_running_stats=track_running_stats,
        bias=bias,
        sp_norm_p=sp_norm_p,
        extra_group_jump=extra_group_jump,
        final_dropout_rate=final_dropout_rate,
    )


def ResNet34(
    num_classes=10,
    shortcut_with_bn=1,
    track_running_stats=False,
    preact=False,
    input_ds="cifar",
    with_classifier=True,
    bias=False,
    sp_norm_p=(0.0, 0.0),
    extra_group_jump=False,
    final_dropout_rate: float = 0.0,
):
    return ResNet(
        Block,
        [3, 4, 6, 3],
        num_classes,
        with_classifier=with_classifier,
        preact=preact,
        input_ds=input_ds,
        shortcut_with_bn=shortcut_with_bn,
        track_running_stats=track_running_stats,
        bias=bias,
        sp_norm_p=sp_norm_p,
        extra_group_jump=extra_group_jump,
        final_dropout_rate=final_dropout_rate,
    )


def ResNet50(
    num_classes=10,
    shortcut_with_bn=1,
    track_running_stats=False,
    preact=False,
    input_ds="cifar",
    with_classifier=True,
    bias=False,
    sp_norm_p=(0.0, 0.0),
    extra_group_jump=False,
    final_dropout_rate: float = 0.0,
):
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes,
        with_classifier=with_classifier,
        preact=preact,
        input_ds=input_ds,
        shortcut_with_bn=shortcut_with_bn,
        track_running_stats=track_running_stats,
        bias=bias,
        sp_norm_p=sp_norm_p,
        extra_group_jump=extra_group_jump,
        final_dropout_rate=final_dropout_rate,
    )


def ResNet101(
    num_classes=10,
    shortcut_with_bn=1,
    track_running_stats=False,
    preact=False,
    input_ds="cifar",
    with_classifier=True,
    bias=False,
    sp_norm_p=(0.0, 0.0),
    extra_group_jump=False,
    final_dropout_rate: float = 0.0,
):
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        num_classes,
        with_classifier=with_classifier,
        preact=preact,
        input_ds=input_ds,
        shortcut_with_bn=shortcut_with_bn,
        track_running_stats=track_running_stats,
        bias=bias,
        sp_norm_p=sp_norm_p,
        extra_group_jump=extra_group_jump,
        final_dropout_rate=final_dropout_rate,
    )


def ResNet152(
    num_classes=10,
    shortcut_with_bn=1,
    track_running_stats=False,
    preact=False,
    input_ds="cifar",
    with_classifier=True,
    bias=False,
    sp_norm_p=(0.0, 0.0),
    extra_group_jump=False,
    final_dropout_rate: float = 0.0,
):
    return ResNet(
        Bottleneck,
        [3, 8, 36, 3],
        num_classes,
        with_classifier=with_classifier,
        preact=preact,
        input_ds=input_ds,
        shortcut_with_bn=shortcut_with_bn,
        track_running_stats=track_running_stats,
        bias=bias,
        sp_norm_p=sp_norm_p,
        extra_group_jump=extra_group_jump,
        final_dropout_rate=final_dropout_rate,
    )


def WideResNet_28_10(
    num_classes=10,
    shortcut_with_bn=1,
    track_running_stats=False,
    preact=False,
    input_ds="cifar",
    with_classifier=True,
    bias=False,
    sp_norm_p=(0.0, 0.0),
    dropout_rate=0.0,
    extra_group_jump=False,
    final_dropout_rate: float = 0.0,
):
    return WideResNet(
        28,
        10,
        num_classes,
        shortcut_with_bn,
        track_running_stats,
        preact=preact,
        input_ds=input_ds,
        with_classifier=with_classifier,
        bias=bias,
        sp_norm_p=sp_norm_p,
        dropout_rate=dropout_rate,
        extra_group_jump=extra_group_jump,
        final_dropout_rate=final_dropout_rate,
    )


def WideResNet_16_8(
    num_classes=200,
    shortcut_with_bn=True,
    track_running_stats=False,
    preact=False,
    input_ds="tiny_imagenet",
    with_classifier=True,
    bias=False,
    sp_norm_p=(0.0, 0.0),
    dropout_rate=0.0,
    extra_group_jump=False,
    final_dropout_rate: float = 0.0,
):
    return WideResNet(
        16,
        8,
        num_classes,
        shortcut_with_bn,
        track_running_stats,
        preact=preact,
        input_ds=input_ds,
        with_classifier=with_classifier,
        bias=bias,
        sp_norm_p=sp_norm_p,
        dropout_rate=dropout_rate,
        extra_group_jump=extra_group_jump,
        final_dropout_rate=final_dropout_rate,
    )
