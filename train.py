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

    EPOCHS = cfg.epochs
    batch_size = cfg.batch
    learning_rate = cfg.learning_rate
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

    val_dataset = LoveWarPlaceDataset(data_set_path=cfg.val_data_path, transforms=transforms_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    if not (train_dataset.class_names == val_dataset.class_names):
        print("error: Numbers of class in training set and test set are not equal")
        exit()

    num_classes = train_dataset.num_classes
    custom_model = CustomConvNet(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(custom_model.parameters(), lr=learning_rate)

    min_val_loss = 10000.0

    for e in range(EPOCHS):
        total_train_loss = 0.0

        for i_batch, item in enumerate(train_loader):
            images = item['image'].to(device)
            labels = item['label'].to(device)

            # Forward pass
            outputs = custom_model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        print('Epoch [{}/{}], train Loss: {:.4f}'.format(e + 1, EPOCHS, total_train_loss / len(train_loader)))

        with torch.no_grad():
            total_val_loss = 0.0

            for i_batch, item in enumerate(val_loader):
                images = item['image'].to(device)
                labels = item['label'].to(device)

                # Forward pass
                outputs = custom_model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

            print('Epoch [{}/{}], val Loss: {:.4f}'
                  .format(e + 1, EPOCHS, total_val_loss / len(val_loader)))

            if total_val_loss / len(val_loader) < min_val_loss:
                min_val_loss = total_val_loss / len(val_loader)
                PATH = os.path.join(cfg.weight_path, "model.pth")
                torch.save(custom_model.state_dict(), PATH)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Place classification in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())

