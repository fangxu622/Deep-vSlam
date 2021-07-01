import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import time

import sys
sys.path.append("/media/fangxu/Segate3T/PHD-Project/VI-Deep-SLAM2")
sys.path.append("G:/PHD-Project/VI-Deep-SLAM2")
from utils.params import par
from utils.helper import normalize_angle_delta,get_data_info

def get_partition_data_info(partition, folder_list, seq_len_range, overlap, sample_times=1, pad_y=False, shuffle=False, sort=True):
    X_path = [[], []]
    Y = [[], []]
    X_len = [[], []]
    df_list = []

    for part in range(2):
        for folder in folder_list:
            start_t = time.time()
            poses = np.load('{}{}.npy'.format(par.pose_dir, folder))  # (n_images, 6)
            fpaths = glob.glob('{}{}/*.png'.format(par.image_dir, folder))
            fpaths.sort()
            # Get the middle section as validation set
            n_val = int((1-partition)*len(fpaths))
            st_val = int((len(fpaths)-n_val)/2)
            ed_val = st_val + n_val
            print('st_val: {}, ed_val:{}'.format(st_val, ed_val))
            if part == 1:
                fpaths = fpaths[st_val:ed_val]
                poses = poses[st_val:ed_val]
            else:
                fpaths = fpaths[:st_val] + fpaths[ed_val:]
                poses = np.concatenate((poses[:st_val], poses[ed_val:]), axis=0)

            # Random Segment
            assert(overlap < min(seq_len_range))
            n_frames = len(fpaths)
            min_len, max_len = seq_len_range[0], seq_len_range[1]
            for i in range(sample_times):
                start = 0
                while True:
                    n = np.random.random_integers(min_len, max_len)
                    if start + n < n_frames:
                        x_seg = fpaths[start:start+n] 
                        X_path[part].append(x_seg)
                        if not pad_y:
                            Y[part].append(poses[start:start+n])
                        else:
                            pad_zero = np.zeros((max_len-n, 6))
                            padded = np.concatenate((poses[start:start+n], pad_zero))
                            Y[part].append(padded.tolist())
                    else:
                        print('Last %d frames is not used' %(start+n-n_frames))
                        break
                    start += n - overlap
                    X_len[part].append(len(x_seg))
            print('Folder {} finish in {} sec'.format(folder, time.time()-start_t))
        
        # Convert to pandas dataframes
        data = {'seq_len': X_len[part], 'image_path': X_path[part], 'pose': Y[part]}
        df = pd.DataFrame(data, columns = ['seq_len', 'image_path', 'pose'])
        # Shuffle through all videos
        if shuffle:
            df = df.sample(frac=1)
        # Sort dataframe by seq_len
        if sort:
            df = df.sort_values(by=['seq_len'], ascending=False)
        df_list.append(df)
    return df_list

class SortedRandomBatchSampler(Sampler):
    def __init__(self, info_dataframe , batch_size, drop_last = False):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.unique_seq_lens = sorted(self.df.iloc[:].seq_len.unique(), reverse=True)
        # Calculate len (num of batches, not num of samples)
        self.len = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            self.len += n_batch

    def __iter__(self):
        # Calculate number of sameples in each group (grouped by seq_len)
        list_batch_indexes = []
        start_idx = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            rand_idxs = (start_idx + torch.randperm(n_sample)).tolist()
            tmp = [rand_idxs[s*self.batch_size: s*self.batch_size+self.batch_size] for s in range(0, n_batch)]
            list_batch_indexes += tmp
            start_idx += n_sample
        return iter(list_batch_indexes)

    def __len__(self):
        return self.len

class ImageSequenceDataset(Dataset):
    def __init__(self, info_dataframe, resize_mode='crop', new_size=None, img_mean=None, img_std=(1,1,1), minus_point_5=False):
        # Transforms
        transform_ops = []
        if resize_mode == 'crop':
            transform_ops.append(transforms.CenterCrop((new_size[0], new_size[1])))
        elif resize_mode == 'rescale':
            transform_ops.append(transforms.Resize((new_size[0], new_size[1])))
        transform_ops.append(transforms.ToTensor())
        #transform_ops.append(transforms.Normalize(mean=img_mean, std=img_std))
        self.transformer = transforms.Compose(transform_ops)
        self.minus_point_5 = minus_point_5
        self.normalizer = transforms.Normalize(mean=img_mean, std=img_std)
        
        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr = np.asarray(self.data_info.image_path)  # image paths
        self.groundtruth_arr = np.asarray(self.data_info.pose)  # pose

    def __getitem__(self, index):
        groundtruth_sequence = torch.FloatTensor((self.groundtruth_arr[index])).reshape(-1,3,4)
        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])  #sequence_len = torch.tensor(len(image_path_sequence))
        image_sequence = []

        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_tensor = self.transformer(img_as_img)
            if self.minus_point_5:
                img_as_tensor = img_as_tensor - 0.5  # from [0, 1] -> [-0.5, 0.5]
            img_as_tensor = self.normalizer(img_as_tensor)
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)

        return (sequence_len, image_sequence, groundtruth_sequence)

    def __len__(self):
        return len(self.data_info.index)

class ImageSequenceDataset_pre(Dataset):
    def __init__(self, info_dataframe, resize_mode='crop', new_sizeize=None, img_mean=None, img_std=(1,1,1), minus_point_5=False):

        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr = np.asarray(self.data_info.image_path)  # image paths
        self.groundtruth_arr = np.asarray(self.data_info.pose)  # pose

    def __getitem__(self, index):
        groundtruth_sequence = torch.FloatTensor((self.groundtruth_arr[index])).reshape(-1,3,4)
        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])  #sequence_len = torch.tensor(len(image_path_sequence))
        
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_tensor = torch.load(img_path).unsqueeze(0)
            image_sequence.append(img_as_tensor)

        image_sequence = torch.cat(image_sequence, 0)

        return (sequence_len, image_sequence, groundtruth_sequence)

    def __len__(self):
        return len(self.data_info.index)

if __name__ == '__main__':
    start_t = time.time()
    # Gernerate info dataframe
    overlap = 1
    sample_times = 3
    folder_list = ['04']
    seq_len_range = [30,30]
    df = get_data_info(folder_list, seq_len_range, overlap, sample_times)
    print('Elapsed Time (get_data_info): {} sec'.format(time.time()-start_t))
    # Customized Dataset, Sampler
    n_workers = 1
    resize_mode = 'crop'
    new_size = (600, 150)
    img_mean = (-0.14968217427134656, -0.12941663107068363, -0.1320610301921484)
    dataset = ImageSequenceDataset(df, resize_mode, new_size, img_mean)
    sorted_sampler = SortedRandomBatchSampler(df, batch_size=4, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=sorted_sampler, num_workers=n_workers)
    print('Elapsed Time (dataloader): {} sec'.format(time.time()-start_t))

    for batch in dataloader:
        s, x, y = batch
        print('='*50)

        print('len:{}\nx:{}\ny:{}'.format(s, x.shape, y.shape))
        break
    
    print('Elapsed Time: {} sec'.format(time.time()-start_t))
    print('Number of workers = ', n_workers)