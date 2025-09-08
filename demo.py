# 测试图片
from ultralytics import YOLO
import cv2
import numpy as np
import sys


# 读取命令行参数
weight_path = './runs/obb_pos/train/weights/best.pt'
media_path = "/data2/sihuo/yolov8_strawberry/0504/combined_data/images/183.png"
media_path = "/data2/sihuo/yolov8_strawberry/0504/video/video4.mp4"
media_path = "/data2/hil_data/yolov8_obb_pos/ultralytics/input/test_val_imgs/"
#media_path = "/data2/sihuo/yolov8_strawberry/0504/video_1/video4.mp4"

# 加载模型
model = YOLO(weight_path)

results = model(media_path, save=True)  # predict on an image

