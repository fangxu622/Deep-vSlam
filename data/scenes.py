import torch
from PIL import Image
import torch.utils.data as data
import quaternion
import os
import numpy as np
import cv2
import glob
from torchvision import transforms

import os.path as osp

SCENES_PATH="/home/t/data/7Scenes/"

class ScenesDataset(data.Dataset):
    def __init__(self, data_dir =SCENES_PATH, train=True, scenes ='all',seq_list=[1,2,3,4], transform_rgb=None, transform_depth = None):
        super(ScenesDataset, self).__init__()

        self.data_dir = data_dir

        if scenes=='all':
            scenes=[d for d in os.listdir(data_dir)]

        self.imgs_path = []
        self.depth_path = []
        self.labels_path = []

        for scene in scenes:
            sceneDir = os.path.join(data_dir+scene)
            seq_list=[]
            if train:
                split_file = osp.join(sceneDir, 'TrainSplit.txt')
            else:
                split_file = osp.join(sceneDir, 'TestSplit.txt')
            with open(split_file, 'r') as f:
                seq_list = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]
            
            for i in seq_list:
                if i<10:
                    seq_idx = "seq-0{}".format(i)
                else:
                    seq_idx = "seq-{}".format(i)

                sequenceDir = os.path.join(data_dir+scene,seq_idx)
                poselabelsNames = glob.glob(sequenceDir+"/*.pose.txt")
                poselabelsNames.sort()
                for label in poselabelsNames:
                    self.labels_path.append( label )
                    self.depth_path.append( label.replace("pose.txt","depth.png") )
                    self.imgs_path.append( label.replace("pose.txt","color.png") )

        self.transform_depth = transform_depth

    def __getitem__(self, index):
        img_color=cv2.imread(self.imgs_path[index])
        pose = np.loadtxt(self.labels_path[index])
        q =  quaternion.from_rotation_matrix(pose[:3,:3] )
        t = pose[:3,3]
        q_arr = quaternion.as_float_array(q)#[np.newaxis,:]
        result = (torch.tensor(img_color).to(torch.float32),torch.tensor(t).to(torch.float32), torch.tensor(q_arr).to(torch.float32) )
        return  result
    
    def __len__(self):
        return len(self.labels_path)


if __name__ == '__main__':
    dataset= ScenesDataset(train=False)
    data_loader = data.DataLoader(dataset, batch_size=1, num_workers=8,shuffle=True)
    for i, data in enumerate(data_loader):
        x,t,q=data
        frame=x[0].numpy()
        cv2.imshow('test',frame.astype(np.uint8))
        print('t:'+str(t))
        print('q:'+str(q))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
        

