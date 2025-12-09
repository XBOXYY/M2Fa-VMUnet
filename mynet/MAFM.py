import torch
import torch.nn as nn
import torch.nn.functional as F

from Experiments.mynet.vmamba import SS2D, S6


class MultiScaleAtrousConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleAtrousConv, self).__init__()

        # 空洞率为1，等价于普通卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False)

        # 空洞率为2
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False)

        # 空洞率为3
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False)

        # 空洞率为5
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=5, dilation=5, bias=False)

        # 最终融合的卷积，用于将所有的多尺度特征整合到一起
        self.fusion_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 使用不同空洞率的卷积
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv1(x)  # 空洞率为1
        x2 = self.conv2(x)  # 空洞率为2
        x3 = self.conv3(x)  # 空洞率为3
        x5 = self.conv5(x)  # 空洞率为5

        # 将不同尺度的输出在通道维度进行拼接
        x_concat = torch.cat((x1, x3, x5), dim=1)

        # 通过 1x1 卷积融合
        x_fused = self.fusion_conv(x_concat)
        # x_fused = x_fused.permute(0, 2, 3, 1)
        x_fused = self.bn(x_fused)
        x_fused = self.relu(x_fused)
        x_fused = x_fused.permute(0, 2, 3, 1)

        return x_fused


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入 x 的形状为 [B, H, W, C]
        # 调整维度为 [B, C, H, W] 以适配卷积操作
        x = x.permute(0, 3, 1, 2)

        # 全局平均池化路径
        avg_out = self.fc(self.avg_pool(x))
        # 全局最大池化路径
        max_out = self.fc(self.max_pool(x))
        # 加和并通过 sigmoid
        out = self.sigmoid(avg_out + max_out)

        # 将注意力权重应用到原始输入
        out = x * out

        # 调整回原始的 [B, H, W, C] 形状
        out = out.permute(0, 2, 3, 1)

        return out


class LSA(nn.Module):
    def __init__(self, kernel_size=7):
        super(LSA, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2  # 保证卷积后输出尺寸一致
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入的形状为 [B, H, W, C]

        # 调整输入维度为 [B, C, H, W] 以适应卷积
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        # 按通道维度进行最大池化和平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化 [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化 [B, 1, H, W]

        # 将两个特征图拼接 [B, 2, H, W]
        out = torch.cat([avg_out, max_out], dim=1)

        # 通过卷积和 sigmoid 计算注意力权重 [B, 1, H, W]
        out = self.sigmoid(self.conv(out))

        # 将注意力权重应用到输入特征
        x = x * out  # [B, C, H, W]

        # 调整回原始输入形状 [B, H, W, C]
        x = x.permute(0, 2, 3, 1)

        return x


class GSA(nn.Module):
    def __init__(self, ):
        super(GSA, self).__init__()
        self.att = S6(d_model=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入的形状为 [B, H, W, C]
        x_t = x
        # 调整输入维度为 [B, C, H, W] 以适应卷积
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        out = torch.mean(x, dim=1, keepdim=True)  # 平均池化 [B, 1, H, W]
        out = out.permute(0, 2, 3, 1)  # [B, H, W, 1]
        # 通过卷积和 sigmoid 计算注意力权重 [B, H, W, 1]
        out = self.sigmoid(self.att(out))

        # 将注意力权重应用到输入特征
        x = x_t * out  # [B, H, W, C]
        return x

class MAFM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(MAFM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.ma = GSA()
        self.sa = LSA()
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # b,h,w,c
        x = self.ca(x)
        x = self.ma(x) + self.sa(x)
        # b,c,h,w
        x = x.permute(0, 3, 1, 2)
        x = self.bn(x)
        x = self.relu(x)
        # b,h,w,c
        x = x.permute(0, 2, 3, 1)
        return x


if __name__ == "__main__":
    # 假设输入的张量形状是 (B, H, W, C) -> (1, 64, 64, 96)
    input_tensor = torch.randn(1, 64, 64, 96).cuda()  # 移动到 GPU

    # 定义 ChannelAttention 和 SS2D 模块
    cama = MAFM(in_channels=96).cuda()

    # 前向传播
    output_tensor = cama(input_tensor)
    print(f"Output Shape: {output_tensor.shape}")
