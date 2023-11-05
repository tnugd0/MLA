from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from torch.utils.data import Dataset
import torch

class Custom_Resnet50(nn.Module):
    def __init__(self, num_class = 43):
        super().__init__()
        self.model = resnet50( weights  = ResNet50_Weights.IMAGENET1K_V2)
        # print(self.model)
        del self.model.fc
        for name , param in self.model.named_parameters():
            param.requires_grad = False


        for param in self.model.layer4.parameters():
            param.requires_grad = True


        for param in self.model.layer3.parameters():
            param.requires_grad = True

        for param in self.model.layer2.parameters():
            param.requires_grad = True

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features = 1024, out_features=512),
            nn.LeakyReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features = 512, out_features=num_class),
        )

    def _forward_impl(self, x):
    # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


# my_retnet = Custom_Resnet50()
# print(my_retnet.model)
# print(self.model)
