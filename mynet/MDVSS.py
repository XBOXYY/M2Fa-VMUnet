import torch
import torch.nn as nn
import torch.nn.functional as F

from Experiments.mynet.MSFE import MSFE
from Experiments.mynet.vmamba import VSSBlock


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入的形状是 (b, h, w, c)，需要先调整为 (b, c, h, w)
        x = x.permute(0, 3, 1, 2)  # 改变维度顺序，从 (b, h, w, c) -> (b, c, h, w)

        # 经过卷积、批量归一化和激活函数
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # 将输出的形状改回 (b, h, w, c)
        x = x.permute(0, 2, 3, 1)  # 改变维度顺序，从 (b, c, h, w) -> (b, h, w, c)
        return x


class ACB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACB, self).__init__()
        # 3x1卷积
        self.conv_3x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn_3x1 = nn.BatchNorm2d(out_channels)  # 对3x1卷积的输出进行Batch Normalization
        # 1x3卷积
        self.conv_1x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn_1x3 = nn.BatchNorm2d(out_channels)  # 对1x3卷积的输出进行Batch Normalization
        # 标准3x3卷积用于补充
        self.conv_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn_3x3 = nn.BatchNorm2d(out_channels)  # 对3x3卷积的输出进行Batch Normalization

        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # 分别通过卷积、BN和ReLU
        out_3x1 = self.relu(self.bn_3x1(self.conv_3x1(x)))
        out_1x3 = self.relu(self.bn_1x3(self.conv_1x3(x)))
        out_3x3 = self.relu(self.bn_3x3(self.conv_3x3(x)))

        # 将各方向的卷积输出相加
        # x = out_3x1 + out_1x3 + out_3x3
        x = out_3x3
        x = x.permute(0, 2, 3, 1)

        return x



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class MDVSS(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, drop_path=0, attn_drop_rate=0, d_state=16, reduction_ratio=16, **kwargs):
        super(MDVSS, self).__init__()
        self.conv_module =ACB(in_channels,out_channels)
        self.vss_block1 = VSSBlock(hidden_dim=hidden_dim, drop_path=drop_path, attn_drop_rate=attn_drop_rate,
                                  d_state=d_state, **kwargs)
    def forward(self, x):
        # 先经过卷积模块
        x = self.conv_module(x)
        x = self.vss_block1(x)
        return x

    # def forward(self, x):
    #     # 先经过卷积模块
    #     conv_out = self.conv_module(x)
    #     # 然后经过通道注意力模块
    #     x = conv_out.permute(0, 3, 1, 2)  # 调整维度顺序以适应通道注意力模块 (b, h, w, c) -> (b, c, h, w)
    #     ca_out = self.channel_attention(x)
    #     # 将通道注意力的输出和卷积输出相加
    #     x = x + ca_out
    #     x = x.permute(0, 2, 3, 1)  # 调整维度顺序回到 (b, h, w, c)
    #     # 最后经过VSSBlock模块
    #     x = self.vss_block(x)
    #     return x
# # 示例用法
if __name__ == "__main__":
    # 假设输入的张量形状是 (b, h, w, c) -> (1, 64, 64, 3)
    input_tensor = torch.randn(1, 64, 64, 3)

    # 定义卷积模块，输入通道数为 3，输出通道数为 16
    conv_module = ConvModule(in_channels=3, out_channels=16)

    # 前向传播
    output_tensor = conv_module(input_tensor)
    print(output_tensor.shape)  # 输出张量的形状应为 (1, 64, 64, 16)
