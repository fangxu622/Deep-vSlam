import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self,loss='mse'):
        super(Loss, self).__init__()
        if loss=='mse':
            self.criterion=nn.MSELoss()
        elif loss=='smoothL1':
            self.criterion=nn.SmoothL1Loss()
        else:
            raise NotImplementedError
        
    def forward(self,pred,target):
        loss=self.criterion(pred,target)
        return(loss)

