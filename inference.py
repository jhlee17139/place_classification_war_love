import argparse
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms

from xml.etree.ElementTree import Element, ElementTree
from utils.utils_config import get_config
from dataset.dataset import LoveWarPlaceDataset
from model.model import CustomConvNet
from tqdm import tqdm
import cv2


def classification_to_xml_format(file_name, category, width, height, depth=3):
    root = Element("annotation")

    folder_anno = Element("folder")
    folder_anno.text = "love_war_frames"
    root.append(folder_anno)

    file_anno = Element("filename")
    file_anno.text = file_name.split('/')[-1]
    root.append(file_anno)

    path_anno = Element("path")
    path_anno.text = "G:\itrc_22\love_war_frames\{}".format(file_name)
    root.append(path_anno)

    source_anno = Element("source")
    database_anno = Element("database")
    database_anno.text = "Love_And_War"
    source_anno.append(database_anno)
    root.append(source_anno)

    size_anno = Element("size")
    width_anno = Element("width")
    width_anno.text = str(width)
    size_anno.append(width_anno)

    height_anno = Element("height")
    height_anno.text = str(height)
    size_anno.append(height_anno)

    depth_anno = Element("depth")
    depth_anno.text = str(depth)
    size_anno.append(depth_anno)
    root.append(size_anno)

    category_anno = Element("category")
    category_anno.text = str(category)
    size_anno.append(category_anno)
    root.append(size_anno)

    tree = ElementTree(root)
    return tree


def main(args):
    # get config
    cfg = get_config(args.config)
    resize = cfg.resize

    transforms_val = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    test_dataset = LoveWarPlaceDataset(data_set_path=cfg.val_data_path, transforms=transforms_val)

    if not (cfg.labels == test_dataset.class_names):
        print("error: Numbers of class in training set and test set are not equal")
        exit()

    class_labels = cfg.labels
    PATH = os.path.join(cfg.weight_path, "model.pth")
    custom_model = CustomConvNet(num_classes=len(class_labels)).to(device)
    custom_model.load_state_dict(torch.load(PATH))
    custom_model.eval()
    img_names = os.listdir(cfg.inference_input)

    for img_name in tqdm(img_names):
        img_path = os.path.join(cfg.inference_input, img_name)
        image = Image.open(img_path)
        image = image.convert("RGB")
        ori_width, ori_height = image.size

        # image = cv2.imread(img_path)
        image = transforms_val(image)
        image = image.to(device)
        image = torch.unsqueeze(image, dim=0)

        outputs = custom_model(image)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.detach().cpu().tolist()[0]
        category = class_labels[predicted]

        tree = classification_to_xml_format(img_path, category, ori_width, ori_height)
        xml_name = os.path.join(cfg.inference_output, '{}.xml'.format(img_path.split('/')[-1].split('.')[0]))

        with open(xml_name, "wb") as file:
            tree.write(file, encoding='utf-8', xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Place classification in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
