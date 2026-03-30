# 注意力可视化快速使用指南

## 快速开始

```bash
# 1. 运行可视化
./run_attention_vis.sh

# 2. 查看图像注意力结果
eog attention_vis/*_image_*.png

# 3. 查看点云注意力结果（PNG预览）
eog attention_vis/*_pointcloud_*.png
```

## 输出文件

### 图像注意力
- `sample{N}_image_query{Q}_layer{L}.png` - 单个query的6个相机视图注意力热力图
- `sample{N}_image_query_merged_layer{L}.png` - **所有query合并的注意力热力图（整体关注区域）**

### 点云注意力（俯视图视角，无坐标轴）
- `sample{N}_pointcloud_query{Q}_layer{L}.png` - 单个query的点云注意力（鸟瞰图PNG）
- `sample{N}_pointcloud_query_merged_layer{L}.png` - **所有query合并的点云注意力（鸟瞰图PNG）**
- `sample{N}_pointcloud_query{Q}_layer{L}.ply` - 完整点云PLY文件
- `sample{N}_pointcloud_keypoints_query{Q}_layer{L}.ply` - 采样的关键点PLY文件
- `sample{N}_pointcloud_query_merged_layer{L}.ply` - 合并的点云PLY文件
- `sample{N}_pointcloud_keypoints_merged_layer{L}.ply` - 合并的关键点PLY文件

## 查看PLY文件

### 方法1: 下载到本地（推荐）
将PLY文件下载到有图形界面的电脑上，用以下软件打开：
- **CloudCompare** - https://www.danielgm.net/cc/
- **MeshLab** - https://www.meshlab.net/

### 方法2: 在服务器上查看（需要X11转发或显示器）
```bash
# 查看单个文件
python tools/view_pointcloud.py attention_vis/sample0_pointcloud_query0_layer5.ply

# 同时查看点云和关键点
python tools/view_pointcloud.py \
    attention_vis/sample0_pointcloud_query0_layer5.ply \
    attention_vis/sample0_pointcloud_keypoints_query0_layer5.ply
```

## 自定义可视化

```bash
python tools/visualize_attention.py \
    work_dirs/0519_grid0075_r50_1600_for_test/height_map_grid0075_with_lightweight_denoise_1600_for_test.py \
    --checkpoint work_dirs/0519_grid0075_r50_1600_for_test/latest.pth \
    --samples 5 \
    --query-ids 0 50 100 \
    --layer-ids 5 \
    --output-dir my_vis
```

### 参数说明
- `--samples N` - 可视化N个样本
- `--query-ids` - 指定查询索引（0-427），不同查询对应不同检测目标
- `--layer-ids` - 指定解码器层（0-5），通常最后一层(5)最清晰
- `--add-merged` - **添加所有query合并的可视化（显示模型整体关注区域）**
- `--output-dir` - 输出目录
- `--save-data` - 保存原始注意力数据为.npz文件

## 理解可视化结果

### 单Query图像注意力
- **红色区域** = 高注意力（模型重点关注的图像区域）
- **蓝色区域** = 低注意力
- **白色圆圈** = 采样的关键点位置
- **用途**: 查看模型如何定位特定物体（一辆车、一个行人等）

### 合并图像注意力（Merged）
- **显示所有query的注意力叠加**
- **红色区域** = 多个query共同关注的重要区域
- **用途**: 了解模型在整个场景中的整体注意力分布
- **与单Query的区别**: 单query看特定物体，merged看整体场景

### 点云注意力（俯视图/鸟瞰图）
- **红色点** = 靠近高注意力关键点
- **蓝色点** = 靠近低注意力关键点
- **灰色点** = 远离所有关键点
- **大彩色点** = 采样的关键点
- **视角**: 俯视图（鸟瞰视角），无坐标轴显示，更适合观察2D投影分布
- **单Query**: 显示特定物体的点云注意力
- **Merged**: 显示所有物体的点云注意力叠加

## 常见问题

### Q: 为什么某些相机视图没有注意力？
A: 如果查询对应的物体不在该相机视角内，采样点会投影到图像外，导致无有效点。尝试其他查询索引。

### Q: 如何知道哪个查询对应哪个物体？
A: 模型有300个查询，前面的查询通常对应更明显的物体。建议多试几个查询索引（0, 50, 100, 150等）。

### Q: 注意力热力图很淡怎么办？
A: 代码已经应用了对比度增强。如果仍然不清晰，可能是该查询的注意力本身就比较分散。

### Q: Open3D窗口报错怎么办？
A: 现在使用matplotlib渲染PNG预览，避免了Open3D窗口问题。PLY文件可以下载到本地用专业软件查看。

## 完整文档

详细使用说明请参考：
- `docs/ATTENTION_VISUALIZATION.md` - 完整使用文档
- `docs/ATTENTION_VIS_SUMMARY.md` - 实现总结
