# 测试图片
from ultralytics import YOLO
import cv2
import numpy as np
import sys
from ultralytics.utils.plotting import output_to_rotated_target, plot_images
from ultralytics.utils import LOGGER, ops



# 读取命令行参数
weight_path = './runs/obb_pos/train/weights/best.pt'
media_path = "/data2/sihuo/yolov8_strawberry/0504/combined_data/images/179.png"

# 加载模型
model = YOLO(weight_path)

results = model.val()  # predict on an image

