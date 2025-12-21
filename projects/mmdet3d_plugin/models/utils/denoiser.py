import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
class HeightMapDenoiser(nn.Module):
    """轻量级高度图去噪模块，使用真实3D边界框作为监督"""
    
    def __init__(self, in_channels=1, hidden_dim=16):
        super(HeightMapDenoiser, self).__init__()
        
        # 轻量级U-Net结构
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(inplace=True)
        )
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_dim*3, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 输出层 - 生成注意力图
        self.outc = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )
        
    def forward(self, x):
        # 编码路径
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        
        # 瓶颈
        b = self.bottleneck(e2)
        
        # 解码路径
        d1 = self.upsample(b)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.dec2(d1)
        
        # 输出logits
        mask_logits = self.outc(d2)
        
        # 在前向传播时使用sigmoid来获得注意力掩码
        mask = torch.sigmoid(mask_logits)
        
        # 应用注意力图到原始高度图
        denoised = x * mask
        
        # 返回denoised和logits
        return denoised, mask_logits
    
class HeightMapDenoiseLoss(nn.Module):
    """基于GT边界框的高度图去噪损失"""
    
    def __init__(self, pc_range, grid_size=0.2, pos_weight=5.0, neg_weight=0.1, loss_weight=1.0):
        super(HeightMapDenoiseLoss, self).__init__()
        self.pc_range = pc_range
        self.grid_size = grid_size
        self.pos_weight = pos_weight  # 增加正样本权重
        self.neg_weight = neg_weight  # 负样本权重
        self.loss_weight = loss_weight  # 总体损失权重
        
    def forward(self, attention_logits, gt_bboxes_3d, height_maps):
        """
        Args:
            attention_logits (Tensor): 注意力掩码logits [B, 1, H, W]
            gt_bboxes_3d (list): GT 3D边界框
            height_maps (Tensor): 原始高度图 [B, 1, H, W]
        
        Returns:
            Tensor: 损失值
        """
        batch_size = height_maps.shape[0]
        device = height_maps.device
        
        # 计算高度图尺寸
        x_size = int((self.pc_range[3] - self.pc_range[0]) / self.grid_size)
        y_size = int((self.pc_range[4] - self.pc_range[1]) / self.grid_size)
        
        total_loss = height_maps.new_tensor(0.0)
        valid_samples = 0
        
        for b in range(batch_size):
            # 生成GT掩码
            gt_mask = torch.zeros((y_size, x_size), device=device)
            
            boxes = gt_bboxes_3d[b].tensor
            if boxes.shape[0] == 0:
                continue
                
            # 转换到BEV坐标系
            cx = ((boxes[:, 0] - self.pc_range[0]) / self.grid_size).float()
            cy = ((boxes[:, 1] - self.pc_range[1]) / self.grid_size).float()
            w = (boxes[:, 3] / self.grid_size).float()
            l = (boxes[:, 4] / self.grid_size).float()
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
                    height_value = boxes[i, 5] / (self.pc_range[5] + 2)  # 归一化高度值，假设最高物体为4米
                    gt_mask[indices[:, 1], indices[:, 0]] = height_value
            
            
            # 可视化gt_mask和pred_b
            try:
                # 确保可视化目录存在
                vis_dir = os.path.join("output_dataset", "0418_norm")
                os.makedirs(vis_dir, exist_ok=True)
                
                # 时间戳和随机标识符以避免文件覆盖
                timestamp = int(time.time() * 1000) % 1000000
                
                # 转换为numpy数组
                gt_np = gt_mask.detach().cpu().numpy()
                
                # 获取原始高度图数据
                orig_np = height_maps[b, 0].detach().cpu().numpy()
                
                """ 
                
                # 应用热力图
                gt_colormap = cm.jet(gt_np)[:, :, :3]  # 取RGB通道
                orig_colormap = cm.jet(orig_np)[:, :, :3] if orig_np is not None else np.zeros_like(gt_colormap)
                
                # 转换为uint8格式用于保存
                gt_colormap = (gt_colormap * 255).astype(np.uint8)
                orig_colormap = (orig_colormap * 255).astype(np.uint8)
                
                # 转换为BGR格式（OpenCV使用BGR）
                gt_colormap_bgr = cv2.cvtColor(gt_colormap, cv2.COLOR_RGB2BGR)
                orig_colormap_bgr = cv2.cvtColor(orig_colormap, cv2.COLOR_RGB2BGR)
                
                # 创建对比图像 - 现在是三部分
                h, w = gt_np.shape
                comparison = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
                comparison[:, :w, :] = orig_colormap_bgr
                comparison[:, w+10:2*w+10, :] = gt_colormap_bgr
                
                # 添加标题
                font = cv2.FONT_HERSHEY_SIMPLEX
                header = np.zeros((30, w*2 + 20, 3), dtype=np.uint8)
                cv2.putText(header, "原始高度图", (w//2-50, 20), font, 0.5, (255,255,255), 1)
                cv2.putText(header, "真实高度图", (w+10+w//2-50, 20), font, 0.5, (255,255,255), 1)
                
                # 合并标题和图像
                final_img = np.vstack([header, comparison])
                
                # 保存图像
                save_path = os.path.join(vis_dir, f"height_map_batch{b}_{timestamp}.png")
                cv2.imwrite(save_path, final_img)
                 """
                
                
                
                # 保存原始数据为numpy文件，方便后续分析
                if orig_np is not None:
                    np.savez(os.path.join(vis_dir, f"{timestamp}.npz"),
                            gt_mask=gt_np, orig=orig_np)
                
            except Exception as e:
                print(f"可视化高度图错误: {e}")
            
            # 分别获取正负样本掩码
            pos_mask = gt_mask > 0
            neg_mask = gt_mask == 0
            
            # 为真实物体区域赋予更高权重
            weight_mask = torch.ones_like(gt_mask)
            weight_mask[pos_mask] = self.pos_weight
            weight_mask[neg_mask] = self.neg_weight
            
            # 只对有点云的区域计算负样本损失
            point_mask = height_maps[b, 0] > 0
            valid_neg_mask = neg_mask & point_mask
            
            # 组合有效掩码
            valid_mask = pos_mask | valid_neg_mask
            
            if valid_mask.any():
                # 使用BCE with logits
                bce_loss = F.binary_cross_entropy_with_logits(
                    attention_logits[b, 0][valid_mask],
                    gt_mask[valid_mask],
                    weight=(weight_mask[valid_mask]).detach()
                )
                
                # Focal Loss with logits
                alpha = 0.25
                gamma = 2.0
                
                # 计算sigmoid后的预测概率
                pred_prob = torch.sigmoid(attention_logits[b, 0][valid_mask])
                p_t = pred_prob * gt_mask[valid_mask] + (1 - pred_prob) * (1 - gt_mask[valid_mask])
                
                # 对正样本使用alpha，对负样本使用(1-alpha)
                alpha_weight = alpha * gt_mask[valid_mask] + (1-alpha) * (1-gt_mask[valid_mask])
                focal_weight = (1 - p_t).pow(gamma) * alpha_weight
                
                focal_loss = F.binary_cross_entropy_with_logits(
                    attention_logits[b, 0][valid_mask],
                    gt_mask[valid_mask],
                    weight=(focal_weight * weight_mask[valid_mask]).detach()
                )
                
                # 结合BCE和Focal Loss
                combined_loss = 0.5 * bce_loss + 0.5 * focal_loss
                total_loss += combined_loss
                valid_samples += 1
        
        if valid_samples > 0:
            return self.loss_weight * (total_loss / valid_samples)
        else:
            return total_loss