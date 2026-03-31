"""
轻量级跨模态融合模块（优化显存）
"""
import torch
import torch.nn as nn



class LightweightCrossModalAttention(nn.Module):
    """
    轻量级跨模态注意力（使用通道注意力，不计算空间注意力）
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 通道注意力（SE 模块）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        # 1x1 卷积融合
        self.fusion_conv = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x1, x2):
        """
        x1: query modality (B, C, H, W)
        x2: key/value modality (B, C, H, W)
        """
        # 拼接
        concat = torch.cat([x1, x2], dim=1)  # (B, 2C, H, W)

        # 通道注意力权重
        attn_weight = self.channel_attn(concat)  # (B, C, 1, 1)

        # 加权 x2
        x2_weighted = x2 * attn_weight

        # 融合
        out = self.fusion_conv(torch.cat([x1, x2_weighted], dim=1))

        return out


class ModalityFusionBlock(nn.Module):
    """
    轻量级三模态融合块
    """

    def __init__(self, dim):
        super().__init__()

        # 跨模态融合（使用轻量级注意力）
        self.rgb_tir_fusion = LightweightCrossModalAttention(dim)
        self.rgb_ms_fusion = LightweightCrossModalAttention(dim)
        self.tir_ms_fusion = LightweightCrossModalAttention(dim)

        # 自适应加权
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 3, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 3, 1),
            nn.Softmax(dim=1)
        )

        # 最终融合
        self.final_conv = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 2, 1),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, rgb_feat, tir_feat, ms_feat):
        """
        rgb_feat, tir_feat, ms_feat: (B, C, H, W)
        Returns: fused_feat (B, C, H, W)
        """
        # 两两融合
        rgb_tir = self.rgb_tir_fusion(rgb_feat, tir_feat)
        rgb_ms = self.rgb_ms_fusion(rgb_feat, ms_feat)
        tir_ms = self.tir_ms_fusion(tir_feat, ms_feat)

        # ���强特征
        rgb_enhanced = rgb_feat + 0.3 * (rgb_tir + rgb_ms)
        tir_enhanced = tir_feat + 0.3 * (rgb_tir + tir_ms)
        ms_enhanced = ms_feat + 0.3 * (rgb_ms + tir_ms)

        # 拼接
        concat = torch.cat([rgb_enhanced, tir_enhanced, ms_enhanced], dim=1)

        # 自适应加权
        weights = self.weight_net(concat)  # (B, 3, 1, 1)
        weighted = weights[:, 0:1] * rgb_enhanced + \
            weights[:, 1:2] * tir_enhanced + \
            weights[:, 2:3] * ms_enhanced

        # 最终融合
        fused = self.final_conv(concat)

        return fused + 0.5 * weighted
