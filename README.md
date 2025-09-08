# 🧠 YOLOv8-OBB-Pose: Dual-Head Model for Oriented Bounding Box Detection and Keypoint Pose Estimation

This project is based on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and extends it with a novel **dual-head architecture** for **oriented bounding box (OBB) detection 📦** and **keypoint-based pose estimation 🎯**. It is especially suited for applications such as strawberry picking, occlusion-aware recognition, and fine-grained object localization.

---

## 📁 Project Structure

This repository consists of the following core components:

- `yolov8_obb_pos_2/`: Main model code, including architecture, training scripts, and inference logic
- `convert_data_tool/`: Utilities for label conversion and annotation visualization
- `labels/`: Example label files using the new unified format
- `result/`: Sample images or video outputs from inference

---

## 🔧 Key Modifications

### ✅ Unified Label Format

Annotation files have been merged to include both OBB and keypoints in the following format:

```
class_id x1 y1 x2 y2 x3 y3 x4 y4 k_x1 k_y1 v1 k_x2 k_y2 v2 k_x3 k_y3 v3 k_x4 k_y4 v4
```

Where:
- The first 8 values represent the 4 corner points of the oriented bounding box
- The remaining 12 values represent 4 keypoints, each with x, y coordinates and a visibility flag \(v \in \{0,1\}\)

---

### ✅ Dataset Input Pipeline

During training, each sample is parsed into:
- `bboxes`: \((x, y, w, h, \theta)\) — representing center point, width, height, and angle
- `keypoints`: \((x_i, y_i, v_i)\) — the position and visibility of each keypoint

---

### ✅ Model Structure

The model retains the YOLOv8 backbone and OBB detection head, and adds a **pose estimation head** on top. The output dimensions are:

```
B × 12 × H × W
```

This means each grid cell predicts 4 keypoints × (x, y, v), totaling 12 channels.

---

## 🚀 Quick Start

### 📄 Step 0: Configure Dataset Path

Edit the file `ultralytics/cfg/datasets/mydata-obb-pose.yaml`:

```yaml
path: ./mydata-obb-pose
```

Make sure this folder exists under `datasets/`, and contains the required `train.txt`, `val.txt`, and `test.txt`.

In your training script, specify the model weight path:

```python
weight_path = './runs/obb_pos/train/weights/best.pt'
```

---

### 🏋️‍♀️ Step 1: Start Training

Run:

```bash
python train.py
```

---

### 👀 Step 2: Visualize Inference Results

Run the demo script to visualize detection and pose:

```bash
python demo.py
```

---

### 📈 Step 3: Evaluate the Model

Run the validation script:

```bash
python val_pos.py
```

Sample output:

```
Class     Images  Instances      Box(P  R  mAP50  mAP50-95)     Pose(P  R  mAP50  mAP50-95)
all         17         28        1   0.963  0.979  0.901      0.882  0.821  0.854  0.764
Speed: 5.0ms preprocess, 4.7ms inference, 0.0ms loss, 10.7ms postprocess per image
```

---

### 🧩 Step 4: Use the Model in Your Own Code

Example inference API usage:

```python
from get_obb_pos import get_obb

obbs = get_obb(weight_path='./runs/obb_pos/train/weights/best.pt', media_path='your_image.jpg')
```

The return is a list of dictionaries, each containing:
- `obbox`: Rotated bounding box
- `keypoints`: Keypoint list
- `is_hidden`: Flag indicating occlusion

---

## 🏷️ Label Format Reference

```
class_id x1 y1 x2 y2 x3 y3 x4 y4  k_x1 k_y1 v1  k_x2 k_y2 v2  k_x3 k_y3 v3  k_x4 k_y4 v4
```

- `x1~x4`, `y1~y4`: Four corner points of the rotated bounding box (clockwise order)
- `k_x`, `k_y`: Keypoint coordinates
- `v`: Visibility flag, where 1 = visible, 0 = occluded

---

## 📸 Visual Results

Check the `./result/` folder for visualization outputs. Keypoints are shown along with bounding boxes:
- ✅ Green dots = visible keypoints
- ❌ Red crosses = occluded keypoints

---

## 📜 License

This project is released under the MIT License — feel free to use, modify, and distribute!

---

## 🙏 Acknowledgements

Built upon [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). Huge thanks to the open-source community!

---

## ⭐ Support This Project

If you find this project useful, please consider giving it a ⭐ star on GitHub — your support keeps it going!
