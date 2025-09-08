# 测试图片
from ultralytics import YOLO
import cv2
import numpy as np
import sys
import torch
import argparse


def get_obb(weight_path, media_path):
    # 加载模型
    model = YOLO(weight_path)
    results = model.predict(source=media_path)

    r = results[0]
    keypoints = r.keypoints.data
    obbs = []
    for i, d in enumerate(reversed(r.obb)):
        box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze()
        result = {}
        result['obb'] = box
        ks = keypoints[i, ...]
        result['is_hidden'] = False
        for j, k in enumerate(ks):
            x_coord, y_coord = k[0], k[1]
            if len(k) == 3:
                conf = k[2]
                if conf < 0.5:
                    result['is_hidden'] = True
                    break
        obbs.append(result)

    return obbs




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="./runs/obb_pos/train/weights/best.pt")
    parser.add_argument('--img', type=str, default="/data2/hil_data/yolov8_obb_pos/ultralytics/input/test_val_imgs/179.png")
    args = parser.parse_args()

    obbs = get_obb(weight_path=args.weight, media_path=args.img)
    for obb in obbs:
        for k, v in obb.items():
            print("name: ", k)
            print("value", v)