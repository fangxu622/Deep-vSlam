import torch
from PIL import Image
import torch.utils.data as data
import quaternion
import os
import numpy as np
import cv2
import glob
from torchvision import transforms

SCENES_PATH="/home/t/data/7Scenes/"

class ScenesDataset(data.Dataset):
    def __init__(self, data_dir =SCENES_PATH, train=True, scenes =["fire/"],seq_list=[1,2,3,4], transform_rgb=None, transform_depth = None):
        super(ScenesDataset, self).__init__()
        imgs_path = []
        depth_path=[]
        labels_path = []

        for scene in scenes:
            sceneDir = os.path.join(data_dir+scene)
            seq_list=[0] # get sequence list
            for d in os.listdir(sceneDir):
                if d[:3]==('seq'):
                    seq_list.append(seq_list[-1]+1)
            seq_list=seq_list[1:]  
             
            if train:  # set last sequence for validation
                seq_list=seq_list[:-1]
            else:
                seq_list=seq_list[-1]

            for i in seq_list:
                if i<10:
                    seq_idx = "seq-0{}".format(i)
                else:
                    seq_idx = "seq-{}".format(i)
                sequenceDir = os.path.join(data_dir+scene,seq_idx)
                poselabelsNames = glob.glob(sequenceDir+"/*.pose.txt")
                poselabelsNames.sort()
                for label in poselabelsNames:
                    labels_path.append( label )
                    depth_path.append( label.replace("pose.txt","depth.png") )
                    imgs_path.append( label.replace("pose.txt","color.png") )

        self.data_dir = data_dir
        self.imgs_path = imgs_path
        self.depth_path = depth_path
        self.transform_depth = transform_depth
        self.labels_path = labels_path

        if transform_depth == None:
            self.transform_depth = transforms.Compose([
                               transforms.ToTensor(),
                               # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                             ])
        else:
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
    dataset= ScenesDataset(scenes=['chess','fire','heads'])
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
        

