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
from Custom_Resnet import Custom_Resnet50
import pandas as pd



def args():
    parser = ArgumentParser()
    parser.add_argument("--image_size", "--i", type=int, default=224, help= "image test size")
    parser.add_argument("--test_image_path_csv", "--tipc",type= str, default= "data/Test.csv", help= "Path to test image" )
    parser.add_argument("--test_image_path", "--tip",type= str, default= "data/Test/00001.png", help= "Path to test image" )
    parser.add_argument("--trained_model_path", "--tmp", type=str, default="./trained_model", help= "trained model store")
    parser.add_argument("--logging", "--l", type=str, default="./runs", help= "tensorboard_path")
    parser.add_argument("--label_path", "--lp", type=str, default="./label.txt", help= "path to list of label")
    parser.add_argument("--image_path", "--ip", type=str, default="None", help= "path to list of label")


    args, unknown = parser.parse_known_args()
    return args




def load_model(model_path,model):
    if model_path:

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
    else:
        print("No model found")
        exit(0)

    return model


def get_label(path):
    with open(path, "r") as file:
        content = file.read().strip().split("\n")


    return content


def read_test_image(root,path):
    
    pf = pd.read_csv(path)
    image_list_path = [os.path.join(root,path) for path in pf["Path"]] 
    class_id_index = [id for id in pf["ClassId"]] 

    return image_list_path , class_id_index

if __name__ == "__main__":

    arg = args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image_list_path , class_id_index = read_test_image("data",arg.test_image_path_csv)

    model = Custom_Resnet50(num_class=43)
    model.to(device)

    categories = get_label(arg.label_path)
   

    load_best_model_path = arg.trained_model_path + "/best_model.pt"

    if os.path.exists(load_best_model_path):
        check_point =  torch.load(load_best_model_path)
        model.load_state_dict(check_point["model"])

    softmax = nn.Softmax()
    count = 0
    correct = 0
    for path, index_id in zip(image_list_path,class_id_index):
        if count == 1000:
            break
        ori_image = cv2.imread(path)
        index = image_list_path.index(path)
        
        image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (arg.image_size, arg.image_size))
        image = np.transpose(image, (2, 0, 1))/255.0
        image = image[None, :,:,:]   # 1 x 3 x 224 x 224

        image = torch.from_numpy(image).to(device).float()

        with torch.no_grad():
            output = model(image)
            probs = softmax(output)
        
        max_idx = torch.argmax(probs)
        
        

        predicted_class = categories[max_idx]
        

        print(max_idx)
        print(predicted_class)

       
       
        if max_idx == index_id:
            print("Predict correct")
            correct += 1
        else:
            print("Predict : " + str(predicted_class))
            print("Actural : " + str(categories[index_id]))

        
        print("The test image is about {} with confident score of {}".format(predicted_class, probs[0, max_idx]))
        print()
        count += 1
        # cv2.imshow("{}:{:.2f}%".format(predicted_class, probs[0, max_idx]*100), ori_image)
        # cv2.waitKey(0)
        # TRAIN & TEST
    print(correct)
    

