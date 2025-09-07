
from ultralytics import YOLO# 加载模型

model = YOLO("ultralytics/cfg/models/v8/yolov8s-obb-pose.yaml", task='obb_pos')  # 从头开始构建新模型# Use the model
results = model.train(data="ultralytics/cfg/datasets/mydata-obb-pose.yaml", epochs=200,device='0')  # 训练模型