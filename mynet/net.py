import torch
from torch import nn

from Experiments.mynet.MDVSS import MDVSS
from Experiments.mynet.PatchExpand import PatchExpand2D, Final_PatchExpand2D
from Experiments.mynet.PatchMerging import PatchMerging2D
from Experiments.mynet.MAFM import MAFM
from Experiments.mynet.MSFE import MSFE
from Experiments.mynet.stem import ConvStem

import torch
import torch.nn as nn


class Fusion(nn.Module):
    def __init__(self, in_channels):
        super(Fusion, self).__init__()

        # 定义 1x1 卷积用于融合拼接的特征
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, enc, expanded):
        x = torch.cat((enc, expanded), dim=3)
        x = x.permute(0, 3, 1, 2)
        # 使用 1x1 卷积融合拼接的特征
        fused = self.conv1x1(x)
        fused = self.bn(fused)
        fused = self.relu(fused)
        fused = fused.permute(0, 2, 3, 1)

        return fused


class M2FaVMUnet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=96):
        super(M2FaVMUnet, self).__init__()
        drop_path_rate = 0.2
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, 4)][::-1]

        # Stem
        self.stem = ConvStem(img_size=224, in_chans=in_channels, embed_dim=base_channels)

        # Encoder
        self.encoder1 = MDVSS(base_channels, base_channels, hidden_dim=base_channels, drop_path=dpr[0])
        self.merge1 = PatchMerging2D(base_channels)
        self.encoder2 = MDVSS(base_channels * 2, base_channels * 2, hidden_dim=base_channels * 2,
                                     drop_path=dpr[1])
        self.merge2 = PatchMerging2D(base_channels * 2)
        self.encoder3 = MDVSS(base_channels * 4, base_channels * 4, hidden_dim=base_channels * 4,
                                     drop_path=dpr[2])
        self.merge3 = PatchMerging2D(base_channels * 4)
        # Neck
        self.necks = nn.ModuleList([
            MSFE(in_channels=base_channels * 8, out_channels=base_channels * 8),
            MDVSS(base_channels * 8, base_channels * 8, hidden_dim=base_channels * 8,
                         drop_path=dpr[3]),
        ])
        # Decoder
        self.expand1 = PatchExpand2D(dim=base_channels * 4)
        self.fusion1 = Fusion(in_channels=base_channels * 4)
        self.decoder1 = MDVSS(base_channels * 4, base_channels * 4, hidden_dim=base_channels * 4,
                                     drop_path=dpr_decoder[1])
        self.expand2 = PatchExpand2D(dim=base_channels * 2)
        self.fusion2 = Fusion(in_channels=base_channels * 2)
        self.decoder2 = MDVSS(base_channels * 2, base_channels * 2, hidden_dim=base_channels * 2,
                                     drop_path=dpr_decoder[2])
        self.expand3 = PatchExpand2D(dim=base_channels)
        self.fusion3 = Fusion(in_channels=base_channels)
        self.decoder3 = MDVSS(base_channels, base_channels, hidden_dim=base_channels, drop_path=dpr_decoder[3])

        # MAFM Modules for skip connections
        self.cama1 = MAFM(in_channels=base_channels * 4)
        self.cama2 = MAFM(in_channels=base_channels * 2)
        self.cama3 = MAFM(in_channels=base_channels)

        # Final convolution
        self.final_up = Final_PatchExpand2D(dim=base_channels, dim_scale=4)
        self.final_conv = nn.Conv2d(base_channels // 4, out_channels, 1)

    def forward(self, x):
        # Stem
        x0 = self.stem(x)

        # Encoder
        enc1 = self.encoder1(x0)  # h/4,w/4,c
        x = self.merge1(enc1)
        enc2 = self.encoder2(x)  # h/8,w/8,2c
        x = self.merge2(enc2)
        enc3 = self.encoder3(x)  # h/16,w/16,4c
        x = self.merge3(enc3)
        for neck in self.necks:
            x = neck(x)
        # Decoder
        d1 = self.decoder1(self.cama1(self.fusion1(enc3, self.expand1(x))))

        d2 = self.decoder2(self.cama2(self.fusion2(enc2, self.expand2(d1))))

        d3 = self.decoder3(self.cama3(self.fusion3(enc1, self.expand3(d2))))
        # Final output
        # out = self.final_up(self.cama4(x0 + d3))
        out = self.final_up(d3)
        out = out.permute(0, 3, 1, 2)
        logits = self.final_conv(out)

        return torch.sigmoid(logits)


# 示例用法
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 假设输入的张量形状是 (b, h, w, c) -> (1, 64, 64, 3)
    input_tensor = torch.randn(1, 3, 256, 256).to(device)  # 将输入张量移动到 GPU
    unet = M2FAVMUnet(in_channels=3, out_channels=1).to(device)

    # 前向传播
    output_tensor = unet(input_tensor)
    print(output_tensor.shape)  # 输出张量的形状
