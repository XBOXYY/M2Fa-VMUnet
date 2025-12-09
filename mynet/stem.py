import torch
from torch import nn


class ConvStem(nn.Module):
    r"""
    Conv Stem for Image Feature Extraction with Downsampling
    Args:
        img_size (int): Input image size. Default: 256.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
    """

    def __init__(self, img_size=224, in_chans=3, embed_dim=96):
        super().__init__()

        # Save input image size
        self.img_size = img_size

        # Define 3x3 convolutions with stride 2 for downsampling
        self.conv1 = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1)  # Output size: img_size / 2
        self.bn1 = nn.BatchNorm2d(embed_dim)  # Batch normalization for conv1

        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)  # Output size: img_size / 4
        self.bn2 = nn.BatchNorm2d(embed_dim)  # Batch normalization for conv2

        # Define a 1x1 convolution layer
        self.conv3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)  # Keep output size the same
        self.bn3 = nn.BatchNorm2d(embed_dim)  # Batch normalization for conv3

        # Compute output size after downsampling
        self.output_size = img_size // 4  # Downsampling by 4

    def forward(self, x):
        # Pass through the first convolution and batch normalization
        x = self.conv1(x)  # (batch_size, embed_dim, img_size / 2, img_size / 2)
        x = self.bn1(x)
        x = nn.ReLU()(x)  # Apply activation

        # Pass through the second convolution and batch normalization
        x = self.conv2(x)  # (batch_size, embed_dim, img_size / 4, img_size / 4)
        x = self.bn2(x)
        x = nn.ReLU()(x)  # Apply activation

        # Pass through the 1x1 convolution and batch normalization
        x = self.conv3(x)  # (batch_size, embed_dim, img_size / 4, img_size / 4)
        x = self.bn3(x)
        x = nn.ReLU()(x)  # Apply activation
        # Change tensor shape for downstream tasks if needed
        x = x.permute(0, 2, 3, 1)

        return x  # Output size: (batch_size, embed_dim, img_size / 4, img_size / 4)

if __name__ == "__main__":
    # 假设输入的张量形状是 (b, h, w, c) -> (1, 64, 64, 3)
    input_tensor = torch.randn(1, 3, 224, 224)

    # 定义卷积模块，输入通道数为 3，输出通道数为 16
    conv_module = ConvStem(img_size=256, in_chans=3, embed_dim=96)

    # 前向传播
    output_tensor = conv_module(input_tensor)
    print(output_tensor.shape)  # 输出张量的形状应为 (1, 64, 64, 16)
