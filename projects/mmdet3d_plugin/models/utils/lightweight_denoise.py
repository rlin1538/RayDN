import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

class DoubleConv(nn.Module):
    """U-Net中的双卷积块"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 输入可能不是整数倍，需要进行裁剪
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出层卷积"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class LightweightDenoiseNet(nn.Module):
    """轻量级点云高度图补全去噪网络"""
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(LightweightDenoiseNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 使用较小的通道数以减小模型体积
        factor = 2 if bilinear else 1
        
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256 // factor)
        
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        
        # 添加一个额外的补全模块
        self.complete_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.complete_out = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # 分支1：分割输出
        segment_out = self.outc(x)
        
        # 分支2：补全输出
        complete_features = F.relu(self.complete_conv(x))
        complete_out = self.complete_out(complete_features)
        
        # 结合两个分支的输出
        final_out = segment_out * torch.sigmoid(complete_out)
        
        return final_out
        
    @staticmethod
    def from_pretrained(pretrained_path):
        """从预训练文件中加载模型
        
        Args:
            pretrained_path (str): 预训练模型的路径
            
        Returns:
            LightweightDenoiseNet: 加载了预训练参数的模型
        """
        model = LightweightDenoiseNet()
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            else:
                model.load_state_dict(checkpoint, strict=True)
            print(f"成功从 {pretrained_path} 加载预训练去噪模型")
        model = model.cuda()
        return model

def denoise_loss(pred, mask, height_map, alpha=0.7, beta=0.3):
    """
    自定义损失函数
    参数:
        pred: 模型预测输出
        mask: 标签mask
        height_map: 原始高度图
        alpha: 分割损失权重
        beta: 补全损失权重
    """
    # # 确保所有输入在有效范围内，避免数值问题
    epsilon = 1e-6
    
    # # 二元交叉熵损失 (分割损失) - 使用更稳定的实现方式
    # # 使用sigmoid确保预测值在[0,1]范围内
    pred_sig = torch.sigmoid(pred)
    
    # # 确保输入严格在[0,1]范围内 - 这是BCE函数的要求
    # # 将值限制在[epsilon, 1-epsilon]范围内，避免极端值
    pred_sig_safe = torch.clamp(pred_sig, epsilon, 1.0 - epsilon)
    mask_safe = torch.clamp(mask, epsilon, 1.0 - epsilon)
    
    # 标准BCE损失
    bce_loss = F.binary_cross_entropy_with_logits(pred_sig_safe, mask_safe, reduction='mean')
    
    # 补全损失 (使用Smooth L1损失)
    # 使用mask的平方作为权重，更关注确定性更高的区域
    mask_weight = mask * mask  # 对mask进行平方，强调正样本区域
    
    # 归一化输入到相似范围，使loss更稳定
    if mask.sum() > epsilon:
        # 只在mask标记的区域内计算smooth_l1损失
        # 使用mask_weight确保只在有效区域计算损失
        pred_normalized = pred_sig * mask_weight
        height_normalized = torch.tanh(height_map) * mask_weight  # 使用tanh归一化高度值，确保在[-1,1]范围内
        
        # 计算损失并加权
        norm_factor = mask_weight.sum() + epsilon  # 添加epsilon避免除以0
        smooth_l1_loss = F.smooth_l1_loss(pred_normalized, height_normalized, reduction='sum') / norm_factor
    else:
        smooth_l1_loss = torch.tensor(0.0, device=pred.device)
    
    # 使用try-except确保数值稳定性
    try:
        # 动态调整两种损失的权重，使它们在数量级上更加平衡
        # 计算损失比例，但添加异常处理
        loss_ratio = (bce_loss / (smooth_l1_loss + epsilon)).item()
        
        # 限制比例在合理范围内，避免极端值
        loss_ratio = max(0.01, min(100.0, loss_ratio))
        
        # 基于比例调整权重，保持总权重为1
        if loss_ratio > 10.0:
            # BCE损失值明显大于smooth_l1损失
            adj_alpha = 0.9
            adj_beta = 0.1
        elif loss_ratio < 0.1:
            # smooth_l1损失明显大于BCE损失
            adj_alpha = 0.1
            adj_beta = 0.9
        else:
            # 使用原始权重
            adj_alpha = alpha
            adj_beta = beta
    except:
        # 如果出现任何数值问题，使用默认权重
        adj_alpha = alpha
        adj_beta = beta
    
    # 确保权重相加等于1
    weight_sum = adj_alpha + adj_beta
    adj_alpha = adj_alpha / weight_sum
    adj_beta = adj_beta / weight_sum
    
    # 结合损失
    total_loss = adj_alpha * bce_loss + adj_beta * smooth_l1_loss
    
    return total_loss#, bce_loss, smooth_l1_loss, adj_alpha, adj_beta

def get_height_mask(gt_bboxes_3d, pc_range, grid_size):
    batch_size = len(gt_bboxes_3d)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # 计算高度图尺寸
    x_size = int((pc_range[3] - pc_range[0]) / grid_size)
    y_size = int((pc_range[4] - pc_range[1]) / grid_size)
    
    gt_masks = torch.zeros((batch_size, y_size, x_size), device=device)
    
    for b in range(batch_size):
        # 生成GT掩码
        # gt_mask = torch.zeros((y_size, x_size), device=device)
        
        boxes = gt_bboxes_3d[b].tensor
        if boxes.shape[0] == 0:
            continue
            
        # 转换到BEV坐标系
        cx = ((boxes[:, 0] - pc_range[0]) / grid_size).float()
        cy = ((boxes[:, 1] - pc_range[1]) / grid_size).float()
        w = (boxes[:, 3] / grid_size).float()
        l = (boxes[:, 4] / grid_size).float()
        theta = boxes[:, 6]
        
        # 计算网格坐标
        y_grid, x_grid = torch.meshgrid(
            torch.arange(y_size, device=device),
            torch.arange(x_size, device=device),
            indexing='ij'
        )
        grid_points = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=1).float()
        
        # 在GT掩码上绘制旋转矩形
        for i in range(len(boxes)):
            # 旋转矩阵
            cos_t = torch.cos(-theta[i])
            sin_t = torch.sin(-theta[i])
            rotation = torch.tensor([[cos_t, sin_t], [-sin_t, cos_t]], device=device)
            
            # 中心点和半宽/半长
            center = torch.tensor([cx[i], cy[i]], device=device)
            half_size = torch.tensor([w[i]/2, l[i]/2], device=device)
            
            # 将点变换到矩形局部坐标系
            local_points = torch.matmul(grid_points - center, rotation)
            
            # 检查点是否在矩形内部
            inside = (local_points[:, 0].abs() <= half_size[0]) & (local_points[:, 1].abs() <= half_size[1])
            
            # 更新GT掩码
            if inside.any():
                indices = grid_points[inside].long()
                height_value = boxes[i, 5] / (pc_range[5] + 2)  # 归一化高度值，假设最高物体为4米
                gt_masks[b, indices[:, 1], indices[:, 0]] = height_value
    
    gt_masks = torch.reshape(gt_masks, (batch_size, 1, y_size, x_size))
    return gt_masks
