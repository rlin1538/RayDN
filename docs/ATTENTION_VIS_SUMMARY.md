# 注意力可视化功能实现总结

## 概述

为RayDN的DETR3D Transformer添加了完整的注意力可视化功能，可以将计算出的注意力权重映射到点云图和6个图像视图中。

## 修改的文件

### 1. projects/mmdet3d_plugin/models/utils/detr3d_transformer.py

修改了两个注意力模块以支持注意力权重保存：

#### DeformableFeatureAggregationCuda 类
- 添加了注意力保存标志和存储变量（第552-557行）
  ```python
  self.save_attention = False
  self.attention_weights_img = None
  self.attention_weights_pts = None
  self.sampled_points_2d = None
  self.sampled_points_3d = None
  ```

- 修改了forward方法，在启用时保存注意力权重（第588-592行，612-613行）
- 修改了feature_sampling方法，返回未归一化的2D采样点坐标（第643-660行）

#### MixedCrossAttention 类
- 添加了相同的注意力保存机制（第718-723行）
- 修改了forward方法保存图像和点云注意力权重（第757-761行，780-782行）
- 修改了feature_sampling_img方法返回2D采样点（第817-835行）

## 新增文件

### 2. tools/attention_visualizer.py

创建了AttentionVisualizer类，提供两个主要可视化方法：

#### visualize_image_attention
- 在6个相机视图上叠加注意力热力图
- 显示采样的关键点，颜色表示注意力权重
- 使用高斯模糊生成平滑的热力图
- 红色=高注意力，蓝色=低注意力

#### visualize_point_cloud_attention
- 根据与关键点的距离和注意力权重给点云着色
- 生成两个文件：完整点云和仅关键点
- 使用Open3D保存和渲染3D可视化

#### save_attention_data
- 保存原始注意力数据为.npz格式供后续分析

### 3. tools/visualize_attention.py

主要的推理脚本，功能包括：

- 加载模型和数据集
- 启用所有注意力模块的权重保存
- 对指定样本执行推理
- 收集指定层的注意力权重
- 为指定的查询生成可视化
- 支持保存原始数据

命令行参数：
- `config`: 配置文件路径（必需）
- `--checkpoint`: 模型权重路径（必需）
- `--samples`: 要可视化的样本数（默认：10）
- `--output-dir`: 输出目录（默认：'attention_vis'）
- `--query-ids`: 要可视化的查询索引（默认：[0, 50, 100]）
- `--layer-ids`: 要可视化的解码器层索引（默认：[5]）
- `--save-data`: 保存原始注意力数据

### 4. docs/ATTENTION_VISUALIZATION.md

完整的使用文档，包括：
- 功能概述
- 依赖项安装
- 使用示例
- 参数说明
- 输出文件格式
- 可视化解读指南
- 工作流示例
- 技术细节
- 故障排除

### 5. run_attention_vis.sh

快速开始脚本，提供一键运行功能：
- 自动检查配置和权重文件
- 激活conda环境
- 使用默认参数运行可视化
- 显示查看结果的命令

## 使用方法

### 快速开始
```bash
# 使用提供的脚本
./run_attention_vis.sh

# 或者直接运行
python tools/visualize_attention.py \
    work_dirs/0519_grid0075_r50_1600_for_test/height_map_grid0075_with_lightweight_denoise_1600_for_test.py \
    --checkpoint work_dirs/0519_grid0075_r50_1600_for_test/latest.pth \
    --samples 5
```

### 高级用法
```bash
python tools/visualize_attention.py \
    work_dirs/0519_grid0075_r50_1600_for_test/height_map_grid0075_with_lightweight_denoise_1600_for_test.py \
    --checkpoint work_dirs/0519_grid0075_r50_1600_for_test/latest.pth \
    --samples 10 \
    --query-ids 0 50 100 150 200 \
    --layer-ids 0 2 5 \
    --save-data \
    --output-dir my_attention_vis
```

## 输出结果

### 图像注意力可视化
- 文件名：`sample{N}_image_query{Q}_layer{L}.png`
- 格式：2×3网格显示6个相机视图
- 内容：注意力热力图叠加在图像上，关键点用圆圈标记

### 点云注意力可视化
- 文件名：
  - `sample{N}_pointcloud_query{Q}_layer{L}.ply` - 完整点云
  - `sample{N}_pointcloud_keypoints_query{Q}_layer{L}.ply` - 仅关键点
  - `sample{N}_pointcloud_query{Q}_layer{L}.png` - 渲染图像

### 原始数据（可选）
- 文件名：`sample{N}_layer{L}_attention_data.npz`
- 包含所有注意力权重和采样点的numpy数组

## 技术细节

### 注意力权重形状
- **图像注意力**: `[B*6, 300, 8, 4*13]`
  - B=1（批次大小）
  - 6个相机视图
  - 300个查询
  - 8个注意力组
  - 4个特征金字塔层级 × 13个采样点

- **点云注意力**: `[B, 300, 13, groups]`
  - B=1（批次大小）
  - 300个查询
  - 13个采样的关键点
  - 注意力组数

### 实现机制
1. 在推理时设置`save_attention=True`标志
2. 注意力模块在forward过程中保存权重到CPU
3. 推理后从模型中收集保存的权重
4. 使用可视化工具处理和渲染数据

## 依赖项

需要额外安装：
```bash
conda activate mv
pip install open3d matplotlib opencv-python
```

## 注意事项

1. **内存使用**：保存注意力权重会增加内存使用，建议一次处理较少的样本
2. **性能**：启用注意力保存会略微降低推理速度
3. **兼容性**：仅在测试/推理时启用，训练时保持关闭
4. **可视化质量**：点云可视化需要Open3D，建议在有图形界面的环境中查看

## 未来改进

可能的扩展方向：
1. 添加自注意力可视化
2. 支持时序注意力可视化
3. 交互式3D可视化界面
4. 注意力统计分析工具
5. 批量处理和对比分析
