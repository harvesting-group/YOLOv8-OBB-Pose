import os
import argparse
import yaml
import cv2

img_width = 0
img_height = 0
key_point_num = 0

class Key_Point:
    def __init__(self, x, y, v):
        self.x = x
        self.y = y
        self.v = v

class Rotate_Box:
    def __init__(self, x1, y1, x2, y2, x3, y3, x4, y4):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4

def get_index_from_file(file_path):
    file_prefix = os.path.splitext(file_path)[-2]
    path_list = file_prefix.split('/', 15)
    return path_list[-1]


def get_list_file(work_dir):
    files = os.listdir(work_dir)
    final_files = []
    index_list = []

    for file in files:
        if os.path.splitext(file)[1] == '.jpg' or os.path.splitext(file)[1] == '.png' or os.path.splitext(file)[1] == '.txt':
            file_path = work_dir + file
            final_files.append(file_path)
            index_list.append(int(get_index_from_file(file_path)))
        else:
            print("File error!")

    print("files size: ", final_files.__len__())
    return final_files, index_list


def read_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def draw_image_for_key_point(image, key_point:Key_Point, color = 0, key_point_id = 0):
    thickness = -1
    bgr = (255, 255, 255)
    if color == 1:
        bgr = (255, 0, 0)
    elif color == 2:
        bgr = (0, 255, 0)
    elif color == 3:
        bgr = (0, 0, 255)
    elif color == 4:
        bgr = (100, 100, 100)
    elif color == 5:
        bgr = (200, 200, 200)

    if key_point_id == 0:
        thickness = -1
    elif key_point_id == 1:
        thickness = 1
    elif key_point_id == 2:
        thickness = 2
    elif key_point_id == 3:
        thickness = 3
    cv2.circle(image, (round(img_width * key_point.x), round(img_height * key_point.y)), 8, bgr, thickness)

def draw_image_for_obb(image, x1, y1, x2, y2, x3, y3, x4, y4, color = 0):
    # print("img_width: ", img_width)
    # print("img_height: ", img_height)
    # print(img_width * x1)
    # print(img_height * y1)
    bgr = (255, 255, 255)
    if color == 1:
        bgr = (255, 0, 0)
    elif color == 2:
        bgr = (0, 255, 0)
    elif color == 3:
        bgr = (0, 0, 255)
    elif color == 4:
        bgr = (100, 100, 100)
    elif color == 5:
        bgr = (200, 200, 200)
    cv2.circle(image, (round(img_width * x1), round(img_height * y1)), 5, bgr, -1)
    cv2.circle(image, (round(img_width * x2), round(img_height * y2)), 5, bgr, -1)
    cv2.circle(image, (round(img_width * x3), round(img_height * y3)), 5, bgr, -1)
    cv2.circle(image, (round(img_width * x4), round(img_height * y4)), 5, bgr, -1)
    cv2.line(image, (round(img_width * x1), round(img_height * y1)), (round(img_width * x2), round(img_height * y2)),
             bgr, 5)
    cv2.line(image, (round(img_width * x2), round(img_height * y2)), (round(img_width * x3), round(img_height * y3)),
             bgr, 5)
    cv2.line(image, (round(img_width * x3), round(img_height * y3)), (round(img_width * x4), round(img_height * y4)),
             bgr, 5)
    cv2.line(image, (round(img_width * x4), round(img_height * y4)), (round(img_width * x1), round(img_height * y1)),
             bgr, 5)

    return image



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', default='', help='yaml path', type=str)
    parser.add_argument('--main_path', default='', help='path to image train path', type=str)
    parser.add_argument('--key_point_image_train_path', default='', help='path to image train path', type=str)
    parser.add_argument('--key_point_label_train_path', default='', help='path to image train path', type=str)
    parser.add_argument('--obb_image_train_path', default='', help='path to image train path', type=str)
    parser.add_argument('--model_path', default='weights/culane_res34.pth',
                        help='path to model file', type=str)
    parser.add_argument('--accuracy', default='fp32', choices=['fp16', 'fp32'], type=str)
    parser.add_argument('--size', default=(800, 320), help='size of original frame', type=tuple)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_yaml_node = read_yaml(args.yaml_path)

    new_image_train_path = config_yaml_node["main_path"] + config_yaml_node["new_image_train_path"]
    new_image_temp_path = config_yaml_node["main_path"] + config_yaml_node["new_image_temp_path"]
    new_label_train_path = config_yaml_node["main_path"] + config_yaml_node["new_label_train_path"]
    path_txt = config_yaml_node["main_path"] + config_yaml_node["path_txt"]
    print("path_txt: ", path_txt)
    start_index = config_yaml_node["start_index"]

    key_point_image_train_file_list, key_point_image_train_file_index = get_list_file(config_yaml_node["main_path"] + config_yaml_node["key_point_image_train_path"])
    key_point_label_train_file_list, key_point_label_train_file_index = get_list_file(config_yaml_node["main_path"] + config_yaml_node["key_point_label_train_path"])

    obb_image_train_file_list, obb_image_train_file_index = get_list_file(config_yaml_node["main_path"] + config_yaml_node["obb_image_train_path"])
    obb_label_train_file_list, obb_label_train_file_index = get_list_file(config_yaml_node["main_path"] + config_yaml_node["obb_label_train_path"])

    train_num = start_index
    demo_img = cv2.imread(key_point_image_train_file_list[0])
    img_width = demo_img.shape[1]
    img_height = demo_img.shape[0]

    print("img_width: ", img_width)
    print("img_height: ", img_height)

    jpg_path_list = []

    for base_index in key_point_image_train_file_index:
        is_match = True
        if base_index not in key_point_label_train_file_index:
            is_match = False
        if base_index not in obb_image_train_file_index:
            is_match = False
        if base_index not in obb_label_train_file_index:
            is_match = False
        if is_match:
            train_num += 1
            index_in_key_point_image_train = key_point_image_train_file_index.index(base_index)
            img = cv2.imread(key_point_image_train_file_list[index_in_key_point_image_train])
            ori_image = img.copy()

            key_point_list_final = []
            rotate_box_list = []

            x1 = 0
            y1 = 0
            x2 = 0
            y2 = 0
            x3 = 0
            y3 = 0
            x4 = 0
            y4 = 0
            x1_zhengkuang = 0
            y1_zhengkuang = 0
            x2_zhengkuang = 0
            y2_zhengkuang = 0

            index_in_obb_label_train = obb_label_train_file_index.index(base_index)
            obb_label_train_file = obb_label_train_file_list[index_in_obb_label_train]
            # print("==================")
            with open(obb_label_train_file, 'r', encoding='utf-8') as file:
                cur_index = 0
                for line in file:
                    single_obb_line = line.strip()
                    coor_list = single_obb_line.split(' ', 25)
                    obb_list = coor_list[1:]
                    x1 = float(obb_list[0])
                    y1 = float(obb_list[1])
                    x2 = float(obb_list[2])
                    y2 = float(obb_list[3])
                    x3 = float(obb_list[4])
                    y3 = float(obb_list[5])
                    x4 = float(obb_list[6])
                    y4 = float(obb_list[7])
                    img = draw_image_for_obb(img, x1, y1, x2, y2, x3, y3, x4, y4, cur_index)
                    rotate_box = Rotate_Box(x1, y1, x2, y2, x3, y3, x4, y4)
                    rotate_box_list.append(rotate_box)
                    cur_index += 1

            index_in_key_point_label_train = key_point_label_train_file_index.index(base_index)
            key_point_label_train_file = key_point_label_train_file_list[index_in_key_point_label_train]

            with open(key_point_label_train_file, 'r', encoding='utf-8') as file:
                cur_index = 0
                for line in file:
                    single_obb_line = line.strip()
                    coor_list = single_obb_line.split(' ', 25)
                    x1_zhengkuang = float(coor_list[1])
                    y1_zhengkuang = float(coor_list[2])
                    x2_zhengkuang = float(coor_list[3])
                    y2_zhengkuang = float(coor_list[4])
                    key_point_list = coor_list[5:]
                    key_point_num = int(key_point_list.__len__() / 3)
                    for key_point_id in range(key_point_num):
                        single_key_point = Key_Point(float(key_point_list[key_point_id * 3 + 0]),
                                                     float(key_point_list[key_point_id * 3 + 1]),
                                                     int(key_point_list[key_point_id * 3 + 2]))
                        draw_image_for_key_point(img, single_key_point, cur_index, key_point_id)
                        key_point_list_final.append(single_key_point)
                    cur_index += 1

            if int(key_point_list_final.__len__() / key_point_num) != int(rotate_box_list.__len__()):
                print("Error in biaozhu!!! Base_index is: ", base_index)
                continue

            save_ori_img_path = new_image_train_path + str(train_num) + ".png"
            jpg_path_list.append(save_ori_img_path)
            cv2.imwrite(save_ori_img_path, ori_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

            save_img_path = new_image_temp_path + str(train_num) + ".png"
            print("save_img_path: ", save_img_path)
            cv2.imwrite(save_img_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

            label_txt_path = new_label_train_path + str(train_num) + ".txt"

            target_num = int(key_point_list_final.__len__() / key_point_num)
            label_path_txt = new_label_train_path + str(train_num) + ".txt"
            with open(label_path_txt, "w") as file:
                for target_id in range(0, target_num):
                    # obb + key_point
                    # target_str = str(0) + " " + str(rotate_box_list[target_id].x1) \
                    #              + " " + str(rotate_box_list[target_id].y1) \
                    #              + " " + str(rotate_box_list[target_id].x2) \
                    #              + " " + str(rotate_box_list[target_id].y2) \
                    #              + " " + str(rotate_box_list[target_id].x3) \
                    #              + " " + str(rotate_box_list[target_id].y3) \
                    #              + " " + str(rotate_box_list[target_id].x4) \
                    #              + " " + str(rotate_box_list[target_id].y4) + " "
                    # for key_point_id in range(0, key_point_num):
                    #     target_str += " " + str(key_point_list_final[target_id * key_point_num + key_point_id].x) \
                    #                   + " " + str(key_point_list_final[target_id * key_point_num + key_point_id].y) \
                    #                   + " " + str(key_point_list_final[target_id * key_point_num + key_point_id].v)
                    # only obb
                    target_str = str(0) + " " + str(rotate_box_list[target_id].x1) \
                                 + " " + str(rotate_box_list[target_id].y1) \
                                 + " " + str(rotate_box_list[target_id].x2) \
                                 + " " + str(rotate_box_list[target_id].y2) \
                                 + " " + str(rotate_box_list[target_id].x3) \
                                 + " " + str(rotate_box_list[target_id].y3) \
                                 + " " + str(rotate_box_list[target_id].x4) \
                                 + " " + str(rotate_box_list[target_id].y4)
                    file.write(target_str + "\n")
            # cv2.imshow("img", img)
            # cv2.waitKey(200)
        # else:
        #     print("Not matched")

    print("Train num is: ", train_num - start_index)

    with open(path_txt, "w") as file:
        for jpg_path in jpg_path_list:
            file.write(jpg_path + "\n")


'''
python create_composite_data.py --yaml_path ./config.yaml
python create_composite_data.py --main_path D:/data/yolov8_strawberry/obb_pose_combined_dataset/ --key_point_image_train_path strawberry_key_point_pose/images/train/ --key_point_label_train_path strawberry_key_point_pose/labels/train/ --obb_image_train_path strawberry_obb/images/train/
'''

