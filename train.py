import torch
import os
import pandas as pd
from torch.utils.data import Dataset,DataLoader, random_split
# from sklearn.model_selection  import train_test_split
from torchvision.transforms import Compose, Resize, ToTensor, ColorJitter,Normalize, RandomHorizontalFlip,RandomVerticalFlip,GaussianBlur,RandomRotation
from PIL import Image
import cv2
import numpy as np
from torchvision.models import resnet50
from argparse import ArgumentParser
from torch.optim import SGD, Adam
import tqdm
import torch.nn as nn
# for running on local

from data_setup import TrafficSign
from Custom_Resnet import Custom_Resnet50

def train_test_split(dataset,train_size):

    train_size = int(train_size * len(dataset))
    test_size = int(len(dataset) - train_size)
    return random_split(dataset,[train_size,test_size])

def args():
    parser = ArgumentParser()
    parser.add_argument("--root", "--r", type=str, default="./data/traffic-sign" ,help= "get the data path")
    parser.add_argument("--epochs", "--e", type=int, default = 30, help= "Number of epochs")
    parser.add_argument("--image_size", "--i", type=int, default=224, help= "image size of input")
    parser.add_argument("--batch_size", "--bs", type=int, default=32, help= "batch size per epoch")

    args, unknown = parser.parse_known_args()
    return args
if __name__ == "__main__":

    arg = args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = Compose([
        Resize((224,224)),
        ColorJitter(
            brightness= (0.6,1),
            contrast= (0.4,0.85)
        ),
        RandomHorizontalFlip(0.3),
        RandomVerticalFlip(0.3),
        # GaussianBlur((3,3)),
        RandomRotation((20)),
        ToTensor()
    ])

    val_transform = Compose([
        Resize((224,224)),
        ToTensor()
    ])

    traffic_sign = TrafficSign(root = "./data/traffic-sign", transform= train_transform)
    train_set, val_set = train_test_split(traffic_sign, train_size=0.8)

    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_transform

    train_dataloader = DataLoader(
        dataset=train_set,
        num_workers=4,
        shuffle= True,
        batch_size = arg.batch_size,
        drop_last=False,
    )

    val_dataloader = DataLoader(
        dataset= val_set,
        num_workers=4,
        shuffle= True,
        batch_size = arg.batch_size,
        drop_last=False,       
    )



    model = Custom_Resnet50()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(lr= 0.01,)

    for epoch in arg.epochs:
        progress_bar = tqdm(train_dataloader)
        model.eval()
        pass
