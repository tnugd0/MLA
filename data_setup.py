import torch
import os
import pandas as pd
from torch.utils.data import Dataset,DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import cv2
import numpy as np


class TrafficSign(Dataset):
    def __init__(self,root, transform= None): 
        self.transform = transform
        df = pd.read_csv(root + "/Train.csv")
        
        data_path = os.path.join(root,"Train")
        self.paths = [os.path.join(root,path) for path in df["Path"]] 
      
        self.labels = df["ClassId"]
        self.classes = [ 
            'Speed limit (20km/h)',
            'Speed limit (30km/h)', 
            'Speed limit (50km/h)', 
            'Speed limit (60km/h)', 
            'Speed limit (70km/h)', 
            'Speed limit (80km/h)', 
            'End of speed limit (80km/h)', 
            'Speed limit (100km/h)', 
            'Speed limit (120km/h)', 
            'No passing', 
            'No passing veh over 3.5 tons', 
            'Right-of-way at intersection', 
            'Priority road', 
            'Yield', 
            'Stop', 
            'No vehicles', 
            'Veh > 3.5 tons prohibited', 
            'No entry', 
            'General caution', 
            'Dangerous curve left', 
            'Dangerous curve right', 
            'Double curve', 
            'Bumpy road', 
            'Slippery road', 
            'Road narrows on the right', 
            'Road work', 
            'Traffic signals', 
            'Pedestrians', 
            'Children crossing', 
            'Bicycles crossing', 
            'Beware of ice/snow',
            'Wild animals crossing', 
            'End speed + passing limits', 
            'Turn right ahead', 
            'Turn left ahead', 
            'Ahead only', 
            'Go straight or right', 
            'Go straight or left', 
            'Keep right', 
            'Keep left', 
            'Roundabout mandatory', 
            'End of no passing', 
            'End no passing veh > 3.5 tons' ]
        

    

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        image_path = self.paths[index]    
        image = Image.open(image_path).convert("RGB")
        if self.transform: 
            image = self.transform(image)
        label = self.labels[index]
        return image, label


if __name__ == "__main__":
    root = "./data/traffic-sign"
    # transform = Compose([
    #     Resize((224,224)),
    #     ToTensor(),
        
    # ])

    traffic = TrafficSign(root,transform= None)
    image , label = traffic.__getitem__(666)
    # index = traffic.classes.index(label)
    image = image.resize((224,244))
    image.show()