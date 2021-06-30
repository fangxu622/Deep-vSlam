
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data.scenes import ScenesDataset
from torch.utils.data import DataLoader
import torch
from model.regressors.PoseNet import PoseNet
from model.losses import Loss


class Solver():
    def __init__(self, model='PoseNet',dataset='7Scenes',config=None):
        '''
        Args:
            model: support PoseNet
            dataset: support 7Scenes
            config:

        Methods: train, test
        '''
        if model=='PoseNet':
            self.model = PoseNet()
            self.print_network(self.model)

        if dataset=='7Scenes':
            training_data= ScenesDataset(train=True)
            self.train_loader = DataLoader(training_data, batch_size=8, num_workers=8,shuffle=True)
            val_data = ScenesDataset(train=False)
            self.val_loader = DataLoader(val_data, batch_size=8 ,num_workers=8, shuffle=True)

        self.model_save_path = 'experiments/%s' % self.model.name

    def print_network(self, model):
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        print('*' * 20)
        print(model.name)
        print(model)
        print('*' * 20)


    def train(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(),
                               lr=0.0003,
                               weight_decay=0.0005)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        num_epochs = 500
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        since = time.time()
        n_iter = 0
        start_epoch = 0
        
        best_train_loss = 10000
        best_val_loss = 10000
        best_train_model = None
        best_val_model = None
        for epoch in range(start_epoch, num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs-1))
            print('-'*20)
            error_train = []
            error_val = []
            for phase in ['train', 'val']:
                loader=self.train_loader
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                    loader=self.val_loader
                for i, data in enumerate(loader):
                    x, t,q = data
                    x=x.permute(0,3,1,2).to(device)
                    pred_t,pred_q = self.model(x)
                    critetrion=Loss()
                    loss_t = critetrion(pred_t,t.to(device))
                    loss_q = critetrion(pred_q,q.to(device))

                    loss = loss_t + loss_q
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        n_iter += 1
                    if (i+1) % 10 == 0:    
                        print('{}th {} Loss: total loss {:.3f} / pos loss {:.3f} / ori loss {:.3f}'.format((i+1), phase, loss, loss_t, loss_q))
                scheduler.step()

            error_train_loss = np.median(error_train)
            error_val_loss = np.median(error_val)

            if (epoch+1) % 50 == 0:
                save_filename = self.model_save_path + '/checkpoint.pth' % epoch
                # save_path = os.path.join('models', save_filename)
                torch.save(self.model.cpu().state_dict(), save_filename)
                if torch.cuda.is_available():
                    self.model.to(device)
            if error_train_loss < best_train_loss:
                best_train_loss = error_train_loss
                best_train_model = epoch
            if error_val_loss < best_val_loss:
                best_val_loss = error_val_loss
                best_val_model = epoch
                save_filename = self.model_save_path + '/best_net.pth'
                torch.save(self.model.cpu().state_dict(), save_filename)
                if torch.cuda.is_available():
                    self.model.to(device)

            print('Train and Validaion error {} / {}'.format(error_train_loss, error_val_loss))
            print('=' * 40)
            print('=' * 40)


    def test(self,test_model_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()
        print('Load pretrained model: ', test_model_path)
        self.model.load_state_dict(torch.load(test_model_path))
        for i, (inputs, poses) in enumerate(self.data_loader):
            print(i)
            inputs = inputs.to(device)


if __name__ == '__main__':

    solver=Solver(model='PoseNet',dataset='7Scenes')
    '''
        Args:
            model: support PoseNet
            dataset: support 7Scenes
            config:

        Methods: 
            train: training , logging , saving weights in experiments/model_name/checkpoint.pth and best_net.pth
            test:
     '''
    solver.train()

    '''
    ToDo: 
        support resume training from checkpoint
    '''