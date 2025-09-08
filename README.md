# 🧠 YOLOv8-OBB-Pose：同时进行方向框检测与关键点姿态估计的双头模型

本项目基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)，引入了创新的**双分支结构**，能够同时完成 **方向性边界框（OBB）检测 📦** 与 **关键点姿态估计 🎯**，特别适用于如草莓采摘、遮挡识别、精细定位等场景。

---

## 📁 项目结构

项目包含以下核心目录：

- `yolov8_obb_pos_2/`：主模型代码，包括网络结构、训练脚本、推理逻辑等
- `convert_data_tool/`：数据标注转换与可视化工具
- `labels/`：转换后的样例标签
- `result/`：部分测试图像与可视化输出结果

---

## 🔧 核心改动说明

### ✅ 统一标签格式

标签文件已将旋转框与关键点统一为如下格式：

```
class_id x1 y1 x2 y2 x3 y3 x4 y4 k_x1 k_y1 v1 k_x2 k_y2 v2 k_x3 k_y3 v3 k_x4 k_y4 v4
```

其中：
- 前 8 项为 OBB 的四个角点坐标；
- 后 12 项为四个关键点的坐标与可见性标志 \( v \in \{0,1\} \)。

---

### ✅ 数据集处理 pipeline

训练数据送入模型时，标签会被解析为：
- `bboxes`: \((x, y, w, h, \theta)\)，表示中心点、宽高与旋转角
- `keypoints`: \((x_i, y_i, v_i)\)，每个关键点坐标与是否可见

---

### ✅ 模型结构调整

基于 YOLOv8-OBB 架构，在原检测头（OBB head）之上增加一个关键点预测分支（pose head），输出维度为：

```
B × 12 × H × W
```

表示每个 grid cell 预测 4 个关键点，每个关键点包括 x、y、v 三个值。

---

## 🚀 快速上手

### 📄 Step 0: 配置数据路径

编辑配置文件 `ultralytics/cfg/datasets/mydata-obb-pose.yaml`：

```yaml
path: ./mydata-obb-pose
```

确保 `mydata-obb-pose` 文件夹位于 `datasets/` 下，并包含以下 3 个 txt 文件（`train.txt`, `val.txt`, `test.txt`）。

在训练脚本中设置模型保存路径：

```python
weight_path = './runs/obb_pos/train/weights/best.pt'
```

---

### 🏋️‍♀️ Step 1: 启动训练

运行：

```bash
python train.py
```

---

### 👀 Step 2: 可视化测试结果

运行以下命令查看检测与关键点可视化效果：

```bash
python demo.py
```

---

### 📈 Step 3: 进行模型评估

运行验证脚本：

```bash
python val_pos.py
```

示例输出：

```
Class     Images  Instances      Box(P  R  mAP50  mAP50-95)     Pose(P  R  mAP50  mAP50-95)
all         17         28        1   0.963  0.979  0.901      0.882  0.821  0.854  0.764
Speed: 5.0ms preprocess, 4.7ms inference, 0.0ms loss, 10.7ms postprocess per image
```

---

### 🧩 Step 4: 关键接口调用

你可以在自己的脚本中这样调用：

```python
from get_obb_pos import get_obb

obbs = get_obb(weight_path='./runs/obb_pos/train/weights/best.pt', media_path='your_image.jpg')
```

返回值为包含以下字段的 `dict` 列表：
- `obbox`: 方向框坐标
- `keypoints`: 关键点列表
- `is_hidden`: 是否存在遮挡关键点

---

## 🏷️ 标签字段说明

```
class_id x1 y1 x2 y2 x3 y3 x4 y4  k_x1 k_y1 v1  k_x2 k_y2 v2  k_x3 k_y3 v3  k_x4 k_y4 v4
```

- OBB：顺时针四个角点的坐标；
- Keypoints：每个关键点位置与可见性；
- 可见性 \(v=0\) 表示遮挡，该点不参与损失计算。

---

## 📸 结果样例

在 `./result/` 文件夹中可查看可视化结果图，标注了旋转框、关键点坐标与遮挡信息。

---

## 📜 License

本项目使用 MIT License 开源，欢迎引用、扩展或改造。

---

## 🙏 鸣谢

本项目基于 Ultralytics YOLOv8 架构开发，感谢其出色的开源贡献！

👉 https://github.com/ultralytics/ultralytics

---

## ⭐ Star 一下吧！

如果你觉得这个项目有帮助，请不要吝啬地点击右上角的 ⭐Star 支持一下！

