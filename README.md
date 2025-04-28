#Project Description
Code Paths:
YOLOv8 algorithm based on OBB+Pose joint improvements: ./yolov8_obb_pos_2

Annotation data conversion tool (with visualization for inspection): ./convert_data_tool

Example of converted label format: ./labels example

Partial video/image test results: ./result

Key Modifications:
Data Enhancement
The original rotated bounding box (OBB) and keypoints are merged into a single annotation file.
The annotation format is as follows:
class_id x1 y1 x2 y2 x3 y3 x4 y4  k_x1 k_y1 v1  k_x2 k_y2 v2  k_x3 k_y3 v3  k_x4 k_y4 v4
...
Dataset Preparation
For model input, the output data includes:

bboxes: filled with rotated bounding boxes (OBB: x, y, w, h, angle)

keypoints: four corner keypoints (k_x1, k_y1, v1, ..., k_x4, k_y4, v4)

Model Construction
Based on the original OBB structure, with the following modifications:

Backbone and head structures remain unchanged.

An additional keypoint head is added.

Model Output
The model now outputs:

The original OBB head predictions.

The additional keypoint head predictions.

Usage Instructions:
Configuration File

Note: When training and validation, you must specify the dataset path in the configuration file ultralytics/cfg/datasets/mydata-obb-pose.yaml.
path: ./mydata-obb-pose
Make sure the folder mydata-obb-pose exists in the datasets/ directory under the project root (yolov8_obb_pos/datasets/mydata-obb-pose), and that it contains three .txt files.

Note: The path to the best model is set in the code as:
weight_path = './runs/obb_pos/train/weights/best.pt'

1.Training
python train.py

2.Visualization
python demo.py

3.Evaluation
python val_pos.py

Sample output:

val: Scanning /data2/hil_data/data_str/combined_data/labels.cache... 17 images, 0 backgrounds, 0 corrupt: 100%|██████████| 17/17 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|█████████

                   all         17         28          1      0.963      0.979      0.901      0.882      0.821      0.854      0.764
Speed: 5.0ms preprocess, 4.7ms inference, 0.0ms loss, 10.7ms postprocess per image
