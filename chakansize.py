import os
from PIL import Image  # 使用Pillow库处理图像文件
import cv2
import json

def read_images_from_folder(folder_path):
    image_files = []
    valid_image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # 图像文件的常见扩展名

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        # 检查文件是否是图像文件
        if any(file_name.lower().endswith(ext) for ext in valid_image_extensions):
            # 构建图像文件的完整路径
            image_file_path = os.path.join(folder_path, file_name)
            # 将图像文件路径添加到列表中
            img = cv2.imread(image_file_path)
            height, width, channels = img.shape
            if width>384:
                image_files.append(image_file_path)

    return image_files

# 用法示例
folder_path = '/media/test/run/count/countx/CounTX-main-arg/FSC147_384_V2/images_384_VarV2'  # 替换为你的文件夹路径
image_files = read_images_from_folder(folder_path)
with open('/media/test/run/count/countx/CounTX-main-arg/LearningToCountEverything-master/data/Train_Test_Val_FSC_147.json') as f:
    fsc172_d_annotations = json.load(f)
test = fsc172_d_annotations['test']
val = fsc172_d_annotations['val']
train = fsc172_d_annotations['train']
my_dict= {}
my_dict["test"]=[]
my_dict["val"]=[]
my_dict["train"]=[]


for i in image_files:
    filename = os.path.basename(i)
    if filename in test:
        my_dict["test"].append(filename)

    elif filename in train:
        my_dict["train"].append(filename)
    elif filename in val:
        my_dict["val"].append(filename)
with open("data.json", "w") as json_file:
    json.dump(my_dict, json_file, indent=4)

