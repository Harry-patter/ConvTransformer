import torch
from torch import nn
from einops import rearrange


# 双卷积
class DoubleConv(nn.Sequential):
    def __init__(self, in_dim, out_dim):
        """
        使用两层3*3卷积提取特征，不改变特征图尺寸
        :param in_dim: 输入特征维度
        :param out_dim: 输出特征维度
        """
        super().__init__(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1, padding_mode='replicate'),
            # nn.GELU(),
            # nn.ReLU(inplace=True),
            nn.InstanceNorm2d(out_dim, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1, padding_mode='replicate'),
            # nn.GELU(),
            # nn.ReLU(inplace=True),
            nn.InstanceNorm2d(out_dim, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            # nn.GELU(),
        )


class EncoderLayer(nn.Sequential):
    def __init__(self, in_dim, out_dim, scalar=2):
        """
        编码隐层
        :param in_dim: 特征维度
        :param out_dim:编码维度
        """
        super().__init__(
            nn.MaxPool2d(scalar, ceil_mode=True),
            DoubleConv(in_dim, out_dim),
        )


class DecoderLayer(nn.Module):
    def __init__(self, in_dim, out_dim, scalar=2):
        """
        解码隐层
        :param in_dim: 特征维度
        :param out_dim: 解码维度
        """
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=scalar, stride=scalar)
        self.double_conv = DoubleConv(in_dim, out_dim)

    def forward(self, y, x):
        y = self.up_sample(y)
        y = y[:, :, :x.shape[-2], :x.shape[-1]]
        y = torch.cat((y, x), dim=1)
        y = self.double_conv(y)

        return y


# class Encoder(nn.Module):
#     def __init__(self, in_dim, out_dim, scalar=2):
#         """
#         解码隐层
#         :param in_dim: 特征维度
#         :param out_dim: 解码维度
#         """
#         super().__init__()
#         self.up_sample = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=scalar, stride=scalar)
#         self.double_conv = DoubleConv(in_dim, out_dim)
#
#     def forward(self, y, x):
#         y = self.up_sample(y)
#         y = y[:, :, :x.shape[-2], :x.shape[-1]]
#         y = torch.cat((y, x), dim=1)
#         y = self.double_conv(y)
#
#         return y


class ConvTransformer(nn.Module):
    def __init__(self, in_dim, feature=64, unet_layer=3, scalar=3, attn_layer=1, head=16, hidden_dim=None):
        """
        结构参照Unet
        :param in_dim: 输入维度
        :param feature: 特征数
        """
        super().__init__()
        self.attn_layer = attn_layer
        self.double_conv = DoubleConv(in_dim, feature)
        self.unet_layer = unet_layer
        self.scalar = scalar
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(self.unet_layer):
            layer_in, layer_out = feature * (2**i), feature * (2**(i+1))
            self.encoder.append(EncoderLayer(layer_in, layer_out, scalar=self.scalar))
            self.decoder.append(DecoderLayer(layer_out, layer_in, scalar=self.scalar))
        # self.sample_down1 = EncoderLayer(feature, feature * 2, scalar=2)
        # self.sample_down2 = EncoderLayer(feature * 2, feature * 4, scalar=2)
        # self.sample_down3 = EncoderLayer(feature * 4, feature * 8, scalar=2)
        # self.sample_down4 = EncoderLayer(feature * 8, feature * 16, scalar=2)
        # self.sample_down5 = EncoderLayer(feature * 16, feature * 32)

        self.transformer = None
        if self.attn_layer > 0:
            emb_size = feature * (2**self.unet_layer)
            if not hidden_dim:
                hidden_dim = emb_size
            transformer_layer = nn.TransformerEncoderLayer(emb_size, head, hidden_dim,
                                                           .0, 'relu', batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(transformer_layer, attn_layer)
        # self.transformer = Transformer(3, feature*16, 8, feature*16, 0.)

        # self.sample_up5 = DecoderLayer(feature * 32, feature * 16)
        # self.sample_up4 = DecoderLayer(feature * 16, feature * 8, scalar=2)
        # self.sample_up3 = DecoderLayer(feature * 8, feature * 4, scalar=2)
        # self.sample_up2 = DecoderLayer(feature * 4, feature * 2, scalar=2)
        # self.sample_up1 = DecoderLayer(feature * 2, feature, scalar=2)

    def forward(self, x):
        x = self.double_conv(x)

        hidden_out = [x,]
        for i in range(self.unet_layer):
            out = self.encoder[i](hidden_out[i])
            hidden_out.append(out)

        # d1 = self.sample_down1(x)
        # d2 = self.sample_down2(d1)
        # d3 = self.sample_down3(d2)
        # d4 = self.sample_down4(d3)
        # y = self.sample_down5(d4)

        # y = d4
        y = hidden_out[-1]
        if self.transformer:
            h, w = y.shape[-2:]
            y = rearrange(y, 'b c h w -> b (h w) c')
            y = self.transformer(y)
            y = rearrange(y, 'b (h w) c -> b c h w', h=h, w=w)

        for i in range(self.unet_layer - 1, -1, -1):
            y = self.decoder[i](y, hidden_out[i])
        # y = self.sample_up5(y, d4)
        # y = self.sample_up4(y, d3)
        # y = self.sample_up3(y, d2)
        # y = self.sample_up2(y, d1)
        # y = self.sample_up1(y, x)

        return y


class Network(nn.Module):
    def __init__(self, hsi_dim, lidar_dim, cls_num, feature=64, **kwargs):
        super().__init__()
        # self.hsi_ct = nn.Sequential(
        #     nn.Conv2d(hsi_dim, 16, 1, bias=False),
        #     ConvTransformer(16, 64)
        # )

        self.hsi_ct = ConvTransformer(hsi_dim, feature, **kwargs)
        self.lidar_ct = ConvTransformer(lidar_dim, feature, **kwargs)
        self.predict = nn.Conv2d(feature*2, cls_num, 1)

    def forward(self, hsi, lidar):
        hsi = self.hsi_ct(hsi)
        lidar = self.lidar_ct(lidar)
        y = self.predict(torch.cat((hsi, lidar), dim=1))
        return y


if __name__ == '__main__':
    x = torch.randn((1, 2, 349, 1905))
    model = ConvTransformer(2)
    y = model(x)
    ...
