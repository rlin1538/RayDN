# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS, build_backbone,build_neck
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
import time
from projects.mmdet3d_plugin.models.utils.denoiser import HeightMapDenoiser, HeightMapDenoiseLoss
from projects.mmdet3d_plugin.models.utils.lightweight_denoise import LightweightDenoiseNet, denoise_loss, get_height_mask
@DETECTORS.register_module()
class RepDetr3D(MVXTwoStageDetector):
    """RepDetr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_frame_head_grads=2,
                 num_frame_backbone_grads=2,
                 num_frame_losses=2,
                 stride=[16],
                 position_level=[0],
                 aux_2d_only=True,
                 single_test=False,
                 pretrained=None,
                 use_pointpillars=False,
                 # height branch
                 use_height_backbone=False,
                 use_height_denoise=False,
                 use_pretrained_denoise=False,
                 pretrained_denoise_path=None,
                 height_backbone=None,
                 height_neck=None,
                 grid_size=0.2,
                 pc_range=None):
        super(RepDetr3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.test_flag = False
        self.use_pointpillars = use_pointpillars
        self.use_height_backbone = use_height_backbone
        self.use_height_denoise = use_height_denoise
        self.use_pretrained_denoise = use_pretrained_denoise
        if use_height_backbone:
            self.height_backbone = build_backbone(height_backbone)
            self.height_neck = build_neck(height_neck)
            self.pc_range = pc_range
            self.grid_size = grid_size
            if use_pretrained_denoise:
                self.denoiser = LightweightDenoiseNet.from_pretrained(pretrained_denoise_path)
            if use_height_denoise:
                # 添加高度图去噪模块
                self.height_denoiser = LightweightDenoiseNet(n_channels=1)
                # self.denoise_loss = HeightMapDenoiseLoss(pc_range=pc_range, grid_size=grid_size)
   

    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        
        img_feats_reshaped = []

        if self.training or training_mode:
            for i in self.position_level:
                BN, C, H, W = img_feats[i].size()
                img_feat_reshaped = img_feats[i].view(B, len_queue, int(BN/B / len_queue), C, H, W)
                img_feats_reshaped.append(img_feat_reshaped)
        else:
            for i in self.position_level:
                BN, C, H, W = img_feats[i].size()
                img_feat_reshaped = img_feats[i].view(B, int(BN/B/len_queue), C, H, W)
                img_feats_reshaped.append(img_feat_reshaped)

        return img_feats_reshaped


    @auto_fp16(apply_to=('pts'), out_fp32=True)
    def extract_pts_feat(self, pts):
        """Extract features of points."""
        if pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        with torch.autocast(device_type='cuda', dtype=torch.half), torch.no_grad():
            x = self.pts_middle_encoder.half()(voxel_features.half(), coors.half(), batch_size)
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        B, C, H, W = x[1].shape
        x = x[1].permute(0, 2, 3, 1).reshape(B, H*W, C)  # [B, H*W, C]
        
        # 生成网格位置编码
        y_pos, x_pos = torch.meshgrid(torch.linspace(0, 1, H, device=x.device),
                            torch.linspace(0, 1, W, device=x.device))
        pos = torch.stack([x_pos, y_pos], dim=-1).reshape(-1, 2)
        pos = pos.unsqueeze(0).repeat(B, 1, 1) # [B, H*W, 2]
        return x, pos
    
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, T, training_mode=False):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, T, training_mode)
        return img_feats

    def obtain_history_memory(self,
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            gt_bboxes=None,
                            gt_labels=None,
                            img_metas=None,
                            centers2d=None,
                            depths=None,
                            gt_bboxes_ignore=None,
                            **data):
        losses = dict()
        T = data['img'].size(1)
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses
        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            for key in data:
                if key in ['instance_inds_2d', 'points', 'pts_feats']:
                    data_t[key] = data[key][i]
                elif key in ['proposals']:
                    data_t[key] = data[key][i]
                elif key == 'img_feats':
                    data_t[key] = [feat[:, i] for feat in data[key]]
                else:
                    data_t[key] = data[key][:, i]

            data_t['img_feats'] = data_t['img_feats']
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True
            loss = self.forward_pts_train(gt_bboxes_3d[i],
                                            gt_labels_3d[i], gt_bboxes[i],
                                            gt_labels[i], img_metas[i], centers2d[i], depths[i],
                                            requires_grad=requires_grad, return_losses=return_losses, **data_t)
            if loss is not None:
                for key, value in loss.items():
                    losses['frame_'+str(i)+"_"+key] = value
        return losses

    def forward_roi_head(self, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(**data)
            return outs_roi

    # def generate_gt_height_map(self, gt_bboxes_3d, pc_range, grid_size=0.2):
    #     """
    #     将3D标注框投影到BEV生成高度图，完全在GPU上操作
    #     gt_bboxes_3d: 3D标注框列表 [LiDARInstance3DBoxes]
    #     pc_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
    #     grid_size: 每个网格的物理尺寸（米）
    #     """
    #     batch_size = len(gt_bboxes_3d)
    #     height_maps = []
    #     device = torch.device('cuda')
        
    #     # 计算BEV图尺寸
    #     x_size = int((pc_range[3] - pc_range[0]) / grid_size)
    #     y_size = int((pc_range[4] - pc_range[1]) / grid_size)
        
    #     for b in range(batch_size):
    #         # 初始化高度图
    #         height_map = torch.zeros((1, y_size, x_size), device=device)
            
    #         # 获取当前帧的3D框
    #         boxes = gt_bboxes_3d[b].tensor
    #         if boxes.shape[0] == 0:
    #             height_maps.append(height_map)
    #             continue
                
    #         # 转换到BEV坐标系
    #         cx = ((boxes[:, 0] - pc_range[0]) / grid_size).float()
    #         cy = ((boxes[:, 1] - pc_range[1]) / grid_size).float()
    #         w = (boxes[:, 3] / grid_size).float()
    #         l = (boxes[:, 4] / grid_size).float()
    #         theta = boxes[:, 6]  # 方向角
            
    #         # 计算网格坐标
    #         y_grid, x_grid = torch.meshgrid(
    #             torch.arange(y_size, device=device),
    #             torch.arange(x_size, device=device),
    #             indexing='ij'
    #         )
    #         grid_points = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=1).float()
            
    #         # 在高度图上绘制旋转矩形
    #         for i in range(len(boxes)):
    #             # 旋转矩阵
    #             cos_t = torch.cos(-theta[i])  # 注意坐标系可能需要反转角度
    #             sin_t = torch.sin(-theta[i])
    #             rotation = torch.tensor([[cos_t, sin_t], [-sin_t, cos_t]], device=device)
                
    #             # 中心点和半宽/半长
    #             center = torch.tensor([cx[i], cy[i]], device=device)
    #             half_size = torch.tensor([w[i]/2, l[i]/2], device=device)
                
    #             # 将点变换到矩形局部坐标系
    #             local_points = torch.matmul(grid_points - center, rotation)
                
    #             # 检查点是否在矩形内部（使用AABB检测）
    #             inside = (local_points[:, 0].abs() <= half_size[0]) & (local_points[:, 1].abs() <= half_size[1])
                
    #             # 更新高度图
    #             flat_indices = inside.nonzero().squeeze(1)
    #             if flat_indices.numel() > 0:
    #                 indices = grid_points[flat_indices].long()
    #                 height_map[0, indices[:, 1], indices[:, 0]] = 1.0
            
    #         # 保存高度图为图片（仅在训练时）
    #         if self.debug and b == 0 and False:  # 仅在训练时保存第一个批次的第一帧
    #             try:
    #                 import cv2
    #                 import numpy as np
                    
    #                 # 转换为numpy数组并缩放到0-255
    #                 height_map_vis = height_map.detach().cpu().numpy()[0] * 255
    #                 height_map_vis = height_map_vis.astype(np.uint8)
                    
    #                 # 创建彩色热力图以便更好地可视化
    #                 height_map_color = cv2.applyColorMap(height_map_vis, cv2.COLORMAP_JET)
                    
    #                 # 保存图片
    #                 save_path = f'vis/height_map_{time.strftime("%Y%m%d_%H%M%S")}.png'
    #                 cv2.imwrite(save_path, height_map_color)
    #             except Exception as e:
    #                 print(f"保存高度图失败: {e}")
                    
    #         height_maps.append(height_map)
        
    #     return torch.stack(height_maps, dim=0)

    def points_to_height_map(self, points):
        """Convert point cloud to height maps from birds-eye-view.
        
        Args:
            points (list[Tensor]): List of point clouds for each sample
        
        Returns:
            Tensor: Height maps [B, H*W, 1]
        """
        B = len(points)
        device = points[0].device
        
        # 获取点云范围
        x_range = (self.pc_range[0], self.pc_range[3])
        y_range = (self.pc_range[1], self.pc_range[4])
        z_range = (self.pc_range[2], self.pc_range[5])
        
        # 设置网格分辨率
        resolution = self.grid_size  # 米/像素
        H = int((y_range[1] - y_range[0]) / resolution)
        W = int((x_range[1] - x_range[0]) / resolution)
        
        batch_height_maps = []
        
        for b in range(B):
            pts = points[b]
            
            # 创建高度图
            height_map = pts.new_zeros((H * W, 1))
            
            # 过滤点云范围
            mask_x = (pts[:, 0] >= x_range[0]) & (pts[:, 0] < x_range[1])
            mask_y = (pts[:, 1] >= y_range[0]) & (pts[:, 1] < y_range[1])
            mask_z = (pts[:, 2] >= z_range[0]) & (pts[:, 2] < z_range[1])
            mask = mask_x & mask_y & mask_z
            
            if mask.sum() > 0:
                valid_pts = pts[mask]
                
                # 转换到图像坐标
                pts_x = ((valid_pts[:, 0] - x_range[0]) / resolution).long()
                pts_y = ((valid_pts[:, 1] - y_range[0]) / resolution).long()
                pts_z = valid_pts[:, 2]
                
                # 计算一维索引
                indices = pts_y * W + pts_x
                
                # 使用unique和max的优化实现
                # start_time = time.time()
                # unique_indices, inverse = torch.unique(indices, return_inverse=True)
                # 方法一：使用bincount+argsort组合
                sorted_z, sort_idx = torch.sort(pts_z, descending=True)
                sorted_indices = indices[sort_idx]
                unique_sorted_indices, unique_idx = torch.unique_consecutive(sorted_indices, return_inverse=True)
                first_occur = torch.cat([torch.ones(1, dtype=torch.bool, device=device),
                                        sorted_indices[1:] != sorted_indices[:-1]])
                max_heights = (sorted_z[first_occur] - z_range[0]) / (z_range[1] - z_range[0])
                # max_heights += 4

                # 添加索引范围检查
                valid_mask = (unique_sorted_indices >= 0) & (unique_sorted_indices < H * W)
                unique_sorted_indices = unique_sorted_indices[valid_mask]
                max_heights = max_heights[valid_mask]

                if len(unique_sorted_indices) > 0:
                    # 使用 clamp 确保索引不越界
                    safe_indices = torch.clamp(unique_sorted_indices, 0, H * W - 1)
                    height_map[safe_indices] = max_heights.unsqueeze(1)

                # 方法二：使用矩阵运算（需要更多内存但更快）
                # index_matrix = F.one_hot(inverse, num_classes=len(unique_indices)).float()
                # max_heights = (index_matrix * pts_z.unsqueeze(1)).max(dim=0)[0]
                # height_map[unique_indices] = max_heights.unsqueeze(1)
                
                # end_time = time.time()
                # print(f"处理第{b}帧点云用时: {end_time - start_time}秒")
            
            # 可视化第一帧的高度图(如果需要)
            if b == 0:
                height_map_2d = height_map.reshape(H, W)
                
                import matplotlib.pyplot as plt
                import numpy as np
                
                plt.figure(figsize=(20,10))
                
                # 子图1: 原始点云投影
                plt.subplot(1, 2, 1)
                # 转换点云坐标到图像像素
                valid_pts = pts[mask]
                pts_x = ((valid_pts[:, 0] - x_range[0]) / resolution).cpu().numpy()
                pts_y = ((valid_pts[:, 1] - y_range[0]) / resolution).cpu().numpy()
                pts_z = valid_pts[:, 2].cpu().numpy()
                pts_z += 4
                
                # 绘制点云散点图
                scatter = plt.scatter(pts_x, H - pts_y, c=pts_z, 
                                    cmap='jet', s=1, vmin=z_range[0] + 4, vmax=z_range[1] + 4)
                plt.colorbar(scatter, label='Height (m)')
                plt.xlim(0, W)
                plt.ylim(0, H)
                # plt.gca().invert_yaxis()  # 保持坐标系方向一致
                plt.title(f'Raw Point Cloud Projection\nPoints: {mask.sum()}')
                plt.xlabel('X (pixels)')
                plt.ylabel('Y (pixels)')
                plt.axis('off')
                plt.grid(alpha=0.3)

                # 子图2: 生成的高度图
                plt.subplot(1, 2, 2)
                height_vis = height_map_2d.cpu().numpy()
                height_mask = height_vis > 0
                
                # 显示高度图
                plt.imshow(np.zeros_like(height_vis))  # 黑色背景
                plt.imshow(np.ma.masked_where(~height_mask, height_vis), cmap='jet', 
                         vmin=0, vmax=1)  # 保持与点云相同的颜色范围
                plt.colorbar(label='Height (m)')
                plt.title('Processed Height Map')
                plt.axis('off')
                
                # 添加对比说明
                plt.suptitle(f'Height Map Comparison (Frame {b})', y=0.98)
                plt.tight_layout()
                plt.savefig('height_comparison.png', bbox_inches='tight', dpi=300)
                plt.close()
            batch_height_maps.append(height_map)
        
        # [B, H*W, 1]
        height_maps = torch.stack(batch_height_maps, dim=0)
        height_maps = height_maps.reshape(B, H, W)
        return height_maps.unsqueeze(1)

    def points_to_height_map_denoised(self, points, gt_bboxes_3d=None, training=False):
        """对点云生成的高度图进行去噪处理"""
        # 生成原始高度图
        height_maps = self.points_to_height_map(points)
        
        if self.use_pretrained_denoise:
            with torch.no_grad():
                height_maps = self.denoiser(height_maps)
            return height_maps, None
        if not self.use_height_denoise:
            return height_maps, None
        
        # 训练时进行去噪并计算损失
        if training and gt_bboxes_3d is not None:
            denoised_maps = self.height_denoiser(height_maps)
            # denoise_loss = self.denoise_loss(attention_mask, gt_bboxes_3d, height_maps)
            height_mask = get_height_mask(gt_bboxes_3d, self.pc_range, self.grid_size)
            denoiser_loss = denoise_loss(denoised_maps, height_mask, height_maps)
            return denoised_maps, denoiser_loss
        # 测试时只进行去噪
        else:
            denoised_maps = self.height_denoiser(height_maps)
            """ # 可视化部分 - 测试时
            try:
                import cv2
                import numpy as np
                import os
                import time
                
                # 确保vis目录存在
                vis_dir = "vis"
                os.makedirs(vis_dir, exist_ok=True)
                
                # 时间戳，用于区分不同帧
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                
                # 遍历batch中的每个样本
                for b in range(height_maps.shape[0]):
                    # 获取原始高度图和去噪后高度图
                    orig_map = height_maps[b, 0].detach().cpu().numpy()
                    denoised_map = denoised_maps[b, 0].detach().cpu().numpy()
                    mask = attention_mask[b, 0].detach().cpu().numpy()
                    
                    # 归一化到0-255用于显示
                    orig_map_norm = np.clip(orig_map / (orig_map.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
                    denoised_map_norm = np.clip(denoised_map / (denoised_map.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
                    mask_norm = np.clip(mask * 255, 0, 255).astype(np.uint8)
                    
                    # 创建彩色热力图
                    orig_map_color = cv2.applyColorMap(orig_map_norm, cv2.COLORMAP_JET)
                    denoised_map_color = cv2.applyColorMap(denoised_map_norm, cv2.COLORMAP_JET)
                    mask_color = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
                    
                    # 合并为一张对比图
                    # 在水平方向上拼接原图、去噪图和掩码图
                    h, w = orig_map_norm.shape
                    header = np.zeros((50, w*3, 3), dtype=np.uint8)
                    
                    # 添加标题
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(header, "Original", (w//2-50, 30), font, 0.7, (255,255,255), 2)
                    cv2.putText(header, "Denoised", (w+w//2-50, 30), font, 0.7, (255,255,255), 2)
                    cv2.putText(header, "Attention Mask", (2*w+w//2-80, 30), font, 0.7, (255,255,255), 2)
                    
                    # 拼接图像
                    comparison = np.vstack([
                        header,
                        np.hstack([orig_map_color, denoised_map_color, mask_color])
                    ])
                    
                    # 保存图像
                    save_path = os.path.join(vis_dir, f"height_comparison_frame{b}_{timestamp}.png")
                    cv2.imwrite(save_path, comparison)
                    
                    # 打印保存路径
                    print(f"已保存高度图对比图：{save_path}")
            except Exception as e:
                print(f"可视化高度图失败: {e}") """
            return denoised_maps, None

    @auto_fp16(apply_to=('points'), out_fp32=True)
    def extract_height_feats(self, points, gt_bboxes_3d=None):
        # height_maps = self.points_to_height_map(points)
        # 根据是否为训练模式调用不同的函数
        if self.training and gt_bboxes_3d is not None:
            height_maps, denoise_loss = self.points_to_height_map_denoised(
                points, gt_bboxes_3d, training=True)
        else:
            height_maps, denoise_loss = self.points_to_height_map_denoised(
                points, training=False)
        # 编码高度图特征
        height_feats = self.height_backbone(height_maps)
        height_feats = self.height_neck(height_feats) # [B, C, H, W]
        B, C, H, W = height_feats[1].shape
        height_feats = height_feats[1].permute(0, 2, 3, 1).reshape(B, H*W, C)  # [B, H*W, C]
        
        # 生成网格位置编码
        y, x = torch.meshgrid(torch.linspace(0, 1, H, device=height_maps.device),
                            torch.linspace(0, 1, W, device=height_maps.device))
        pos = torch.stack([x, y], dim=-1).reshape(-1, 2)
        pos = pos.unsqueeze(0).repeat(B, 1, 1) # [B, H*W, 2]
        # 如果是训练模式，返回去噪损失
        if self.training and denoise_loss is not None:
            return height_feats, pos, denoise_loss
        else:
            return height_feats, pos

    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """

        if self.use_height_backbone:
            # data['pts_feats'], data['pts_pos'] = self.extract_height_feats(data['pts_feats'], gt_bboxes_3d)
            if self.training and not self.use_pretrained_denoise:
                data['pts_feats'], data['pts_pos'], denoise_loss = self.extract_height_feats(
                    data['pts_feats'], gt_bboxes_3d)
            else:
                data['pts_feats'], data['pts_pos'] = self.extract_height_feats(data['pts_feats'])
        elif self.use_pointpillars:
            data['pts_feats'], data['pts_pos'] = self.extract_pts_feat(data['pts_feats'])

        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(img_metas, **data)
            self.train()

        else:
            outs_roi = self.forward_roi_head(**data)
            outs = self.pts_bbox_head(img_metas, **data)

        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
            if self.with_img_roi_head:
                loss2d_inputs = [gt_bboxes, gt_labels, centers2d, outs_roi, depths, img_metas]
                losses2d = self.img_roi_head.loss(*loss2d_inputs)
                losses.update(losses2d) 
            # 添加去噪损失
            if self.use_height_backbone and self.training and not self.use_pretrained_denoise and denoise_loss is not None:
                losses['height_denoise_loss'] = denoise_loss

            return losses
        else:
            return None

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        for key in ['proposals', 'instance_inds_2d', 'points']:
            if key in data:
                data[key] = list(zip(*data[key]))
        if return_loss:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
                data[key] = list(zip(*data[key]))
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      **data):
        T = data['img'].size(1)

        prev_img = data['img'][:, :-self.num_frame_backbone_grads]
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]
        rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads)

        if T - self.num_frame_backbone_grads > 0:
            self.eval()
            with torch.no_grad():
                prev_img_feats = self.extract_feat(prev_img, T - self.num_frame_backbone_grads, True)
            self.train()
            data['img_feats'] = torch.cat([prev_img_feats, rec_img_feats], dim=1)
        else:
            data['img_feats'] = rec_img_feats
        if self.use_height_backbone or self.use_pointpillars:
            data['pts_feats'] = data['points']

        losses = self.obtain_history_memory(gt_bboxes_3d,
                        gt_labels_3d, gt_bboxes,
                        gt_labels, img_metas, centers2d, depths, gt_bboxes_ignore, **data)

        return losses
  
  
    def forward_test(self, img_metas, rescale, **data):
        self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key != 'img':
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        outs_roi = self.forward_roi_head(**data)

        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = data['img'].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = data['img'].new_ones(1)
        if self.use_height_backbone:
            data['pts_feats'], data['pts_pos'] = self.extract_height_feats(data['points'])
        if self.use_pointpillars:
            data['pts_feats'], data['pts_pos'] = self.extract_pts_feat(data['points'])

        outs = self.pts_bbox_head(img_metas, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        data['img_feats'] = self.extract_img_feat(data['img'], 1)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
    
    # def forward_dummy(self,
    #                   points=None,
    #                   img_metas=None,
    #                   img_inputs=None,
    #                   **kwargs):
    #     img_feats = self.extract_feat(
    #         img=img_inputs, 1, **kwargs)
    #     assert self.with_pts_bbox
    #     outs = self.pts_bbox_head(img_feats)
    #     return outs