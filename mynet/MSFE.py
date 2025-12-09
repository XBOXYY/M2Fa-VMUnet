import torch
import torch.nn as nn
import torch.nn.functional as F


class DDConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2):
        super().__init__()
        # 计算保持尺寸所需的padding
        padding = dilation if kernel_size == 3 else dilation * (kernel_size - 1) // 2

        # 深度空洞卷积
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # 深度卷积
            bias=False
        )

        # 可选：添加批归一化和激活
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # 逐点卷积（1×1卷积）
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return x

class MSFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSFE, self).__init__()

        # 空洞率为1，等价于普通卷积
        self.conv1 = DDConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation=1
        )
        # 空洞率为2
        self.conv2 = DDConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation=2
        )
        # 空洞率为3
        self.conv3 = DDConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation=3
        )
        # 空洞率为5
        self.conv5 = DDConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation=5
        )

        # 最终融合的卷积，用于将所有的多尺度特征整合到一起
        self.fusion_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 使用不同空洞率的卷积
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv1(x)  # 空洞率为1
        x2 = self.conv2(x)  # 空洞率为2
        x3 = self.conv3(x) # 空洞率为3
        x5 = self.conv5(x)  # 空洞率为5

        # 将不同尺度的输出在通道维度进行拼接
        x_concat = torch.cat((x1,x2,x3, x5), dim=1)

        # 通过 1x1 卷积融合
        x_fused = self.fusion_conv(x_concat)
        # x_fused = x_fused.permute(0, 2, 3, 1)
        x_fused = self.bn(x_fused)
        x_fused = self.relu(x_fused)
        x_fused = x_fused.permute(0, 2, 3, 1)

        return x_fused

