# 🎯 YOLOv8-OBB-Pose: Rotated Bounding Box + Keypoint Estimation

This project extends the YOLOv8 framework with joint **Oriented Bounding Box (OBB)** and **Keypoint (Pose)** detection capabilities. It is designed for applications requiring both rotated object localization and fine-grained pose estimation (e.g., strawberries, robotic grasping, aerial imagery).

---

## 📁 Project Structure

```text
./yolov8_obb_pos_2         # YOLOv8 model with OBB + Pose head
./convert_data_tool        # Annotation converter and visualizer
./labels example           # Example of converted annotation format
./result                   # Sample visual/video results
```

---

## 🔧 Annotation Format & Data Enhancement

The original rotated bounding boxes (OBB) and keypoints are unified into a single label file with the following format:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4 k_x1 k_y1 v1 k_x2 k_y2 v2 k_x3 k_y3 v3 k_x4 k_y4 v4 ...
```

- `x1~x4, y1~y4`: Coordinates of the four corners (OBB)
- `k_xi, k_yi, vi`: Keypoint (x, y) and visibility
  - `vi` ∈ {0: not visible, 1: unclear, 2: visible}

These are parsed and used in model training as:

```text
bboxes    → rotated box parameters (x, y, w, h, angle)
keypoints → four corner points with visibility flags
```

---

## 🧱 Model Architecture Modifications

- 🧩 **Backbone and detection heads** are inherited from YOLOv8
- ➕ **Added a dedicated keypoint head** for pose estimation

### 📤 Model Outputs:

- OBB head (rotated box)
- Pose head (keypoint heatmaps or coordinates)

---

## ⚙️ Dataset Configuration

Edit the dataset config YAML file:

```yaml
# File: ultralytics/cfg/datasets/mydata-obb-pose.yaml
path: ./mydata-obb-pose
```

Dataset directory should look like:

```text
datasets/
└── mydata-obb-pose/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Set the model weight path in code:

```python
weight_path = './runs/obb_pos/train/weights/best.pt'
```

---

## 🚀 Quick Start

```bash
# 1. Train the model
python train.py

# 2. Visualize predictions
python demo.py

# 3. Evaluate performance
python val_pos.py
```

---

## 📊 Sample Evaluation Output

```text
val: Scanning /data/.../labels.cache...
17 images, 28 instances

Class     Images  Instances  Box(P R mAP50 mAP50-95)  Pose(P R mAP50 mAP50-95)
all           17         28    0.963  0.979  0.901  0.882    0.821  0.854  0.764

Speed: 5.0ms preprocess, 4.7ms inference, 0.0ms loss, 10.7ms postprocess per image
```

---

## 📎 Notes

- ✅ The `convert_data_tool` can be used to transform original annotations into the required OBB + Pose format, with visualization support.
- ✅ The pose head currently supports four keypoints, but can be extended to more depending on your application.
- ✅ You can adapt this framework for different rotated object scenarios (e.g., leaves, tools, drones).

---

## 📬 Contact

If you find this project useful or have questions, feel free to open an issue or submit a pull request. Contributions are welcome!

