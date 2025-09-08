# 测试图片
from ultralytics import YOLO
import cv2
import numpy as np
import sys
from ultralytics.utils.plotting import output_to_rotated_target, plot_images
from ultralytics.utils import LOGGER, ops
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch

class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'
kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

# 读取命令行参数
weight_path = './runs/obb_pos/train/weights/best.pt'
media_path = "/data2/sihuo/yolov8_strawberry/0504/combined_data/images/180.png"

# 加载模型
model = YOLO(weight_path)
results = model.predict(source=media_path)
im_save = cv2.imread(media_path)
radius=5
kpt_line=True

for r in results:
    print(r.keypoints)

    # this line is changed
    # keypoints_xy = r.keypoints.xy.cpu().int().numpy()  # get the keypoints
    # keypoints_conf = r.keypoints.conf.cpu().numpy()

    # rbox = r.obb
    for k in reversed(r.keypoints[0].data):
        #annotator.kpts(k, self.orig_shape, radius=kpt_radius, kpt_line=kpt_line)
        nkpt, ndim = k.shape
        # is_pose = nkpt == 17 and ndim in {2, 3}
        is_pose = nkpt == 2 and ndim in {2, 3}
        # kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
        for i, k in enumerate(k):
            color_k = [int(x) for x in kpt_color[i]] if is_pose else colors(i)
            x_coord, y_coord = k[0], k[1]
            # if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < 0.25:
                    continue
            cv2.circle(im_save, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

    for d in reversed(r.obb[0]):
        # c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
        # name = ("" if id is None else f"id:{id} ") + names[c]
        # label = (f"{name} {conf:.2f}" if conf else name) if labels else None
        box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze()
        #annotator.box_label(box, label, color=colors(c, True), rotated=is_obb)
        if isinstance(box, torch.Tensor):
            box = box.tolist()
            p1 = [int(b) for b in box[0]]
            # NOTE: cv2-version polylines needs np.asarray type.
            cv2.polylines(im_save, [np.asarray(box, dtype=int)], True, (0, 0, 255), 2)



    cv2.imwrite("test.jpg", im_save)

#change the keypoints order and no.of keypoints accordingly
#As keypoints returns a tensor in r.keypoints we convert extract the kptpoints