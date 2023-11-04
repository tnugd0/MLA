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
from tqdm.autonotebook import tqdm
import torch.nn as nn
import shutil
from sklearn.metrics import accuracy_score,confusion_matrix
from torch.utils.tensorboard import SummaryWriter
# for running on local

import matplotlib.pyplot as plt

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
    parser.add_argument("--batch_size", "--bs", type=int, default=8, help= "batch size per epoch")
    parser.add_argument("--trained_model_path", "--tmp", type=str, default="./trained_model", help= "trained model store")
    parser.add_argument("--logging", "--l", type=str, default="./runs", help= "tensorboard_path")



    args, unknown = parser.parse_known_args()
    return args




def check_stored_path(trained_path, loggin_path):
    if not os.path.exists(trained_path):
        os.mkdir(trained_path)

    if os.path.exists(loggin_path):
        shutil.rmtree(loggin_path)

    
def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def load_model(model_path,model,optimizer):
    if not os.path.isfile(model_path):
        best_acc = -1
        start_epoch =0 
    else:
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_score"]
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint["optimizer"])

    return start_epoch, best_acc,model,optimizer


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
    train_set, val_set = train_test_split(traffic_sign, train_size=0.9)

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

    check_stored_path(arg.trained_model_path, arg.logging)



    # define loss and optimizer
    model = Custom_Resnet50()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= 0.001, eps=1e-07, weight_decay=0.5, amsgrad= True)

    progress_bar = tqdm(train_dataloader, colour="cyan")
    writer = SummaryWriter(arg.logging)

    load_model_path = arg.trained_model_path + "/last_model.pt"

    start_epoch, best_acc,model,optimizer = load_model(load_model_path,model,optimizer)

    # TRAIN & TEST
    for epoch in range(start_epoch,arg.epochs):
        
        model.train()
        
        
        for iter, (image,label) in enumerate(progress_bar):
            image = image.to(device)
            label = label.to(device)
            prediction = model(image)

            loss = criterion(prediction, label)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch+1, arg.epochs, iter+1, len(train_dataloader), loss))
            writer.add_scalar("Train/Loss", loss , epoch*len(train_dataloader) + iter)
            optimizer.zero_grad() #clear the buffer
            loss.backward() # backpro
            optimizer.step() # update weights
        

        model.eval()

        all_predictions = []
        all_labels = []

        best_score = -1

        for iter, (image,label) in enumerate(val_dataloader):
            all_labels.extend(label)
            image = image.to(device)
            label = label.to(device)



            with torch.no_grad():
                
                prediction = model(image)
                print(prediction.shape)

                index = torch.argmax(prediction, dim= 1)
                all_predictions.extend(index)
            
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]

        accuracy = accuracy_score(all_labels,all_predictions)

        # plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), class_names = val_set.classes, epoch=epoch)

        print("Epoch {}: Accuracy: {}".format(epoch+1, accuracy))
        writer.add_scalar("Val/Accuracy", accuracy, epoch)


        # save best trained model
        if best_score < accuracy:

            best_score = accuracy
            checkpoint = {
                "epoch" : epoch + 1,
                "model" : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "best_score": best_score
            }

            torch.save(checkpoint, "{}/best_model.pt".format(arg.trained_model_path))
            best_score = accuracy


        # save the latest trained model

        checkpoint = {
            "epoch" : epoch + 1,
            "model" : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "best_score": best_score
        }
        torch.save(checkpoint, "{}/last_model.pt".format(arg.trained_model_path))
