import argparse
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms

from utils.utils_config import get_config
from dataset.dataset import LoveWarPlaceDataset
from model.model import CustomConvNet


def main(args):
    # get config
    cfg = get_config(args.config)
    batch_size = cfg.batch
    resize = cfg.resize

    transforms_train = transforms.Compose(
        [transforms.Resize((resize, resize)),
         transforms.RandomRotation(cfg.random_rotate),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    transforms_val = transforms.Compose([transforms.Resize((resize, resize)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    train_dataset = LoveWarPlaceDataset(data_set_path=cfg.train_data_path, transforms=transforms_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = LoveWarPlaceDataset(data_set_path=cfg.val_data_path, transforms=transforms_val)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    if not (train_dataset.class_names == test_dataset.class_names):
        print("error: Numbers of class in training set and test set are not equal")
        exit()

    num_classes = train_dataset.num_classes
    labels_name = train_dataset.class_names
    PATH = os.path.join(cfg.weight_path, "model.pth")
    custom_model = CustomConvNet(num_classes=num_classes).to(device)
    custom_model.load_state_dict(torch.load(PATH))

    with torch.no_grad():
        total_cnt = 0
        total_accuracy = 0

        labels_length = len(labels_name)
        labels_correct = list(0. for i in range(labels_length))
        labels_total = list(0. for i in range(labels_length))
        per_label_accuracy = list(0. for i in range(labels_length))

        for test_idx, item in enumerate(test_loader):
            img = item['image'].to(device)
            local_labels = item['label'].to(device)
            outputs = custom_model(img)
            _, predicted = torch.max(outputs, 1)

            total_cnt += outputs.size(0)
            total_accuracy += (predicted == local_labels).sum().item()

            # per class eval
            label_correct_running = (predicted == local_labels).squeeze()

            label_list = local_labels.cpu().tolist()
            label_correct_list = label_correct_running.cpu().tolist()

            for correct_idx in range(len(label_list)):
                if label_correct_list[correct_idx]:
                    labels_correct[label_list[correct_idx]] += 1

                labels_total[label_list[correct_idx]] += 1

        mean_accuracy = total_accuracy / total_cnt
        print('total accuracy : {}'.format(mean_accuracy))

        for i in range(labels_length):
            per_label_accuracy[i] = labels_correct[i] / labels_total[i]
            print('labels {} accuracy : {}'.format(labels_name[i], per_label_accuracy[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Place classification in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
