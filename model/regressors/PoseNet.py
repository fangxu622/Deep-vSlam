import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class PoseNet(nn.Module):
    def __init__(self, base_model='vgg16', transfer=True, dropout_rate=0.0):
        super(PoseNet, self).__init__()
        if base_model=='vgg16':
            self.model=models.vgg16(pretrained=False,num_classes=7)
        self.name='PoseNet'
        if transfer:
            base_model= models.vgg16(pretrained=True)
            self.model.features=base_model.features
            for param in self.model.features:
                param.requires_grad = False

    def forward(self, x):
        x=self.model(x)
        pre_t=x[:,:3]
        pred_q=x[:,3:]
        return pre_t,pred_q

if __name__ == '__main__':
    model=PoseNet()
    print(model)