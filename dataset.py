from PIL import Image
from torch.utils import data
import transforms as trans
from torchvision import transforms
import random
import os
import numpy as np
import cv2
import torch 

def get_loader(dataset_list, data_root, alternate, len_snippet, img_size, mode='train'):

    if mode == 'train' or mode == 'val':

            img_transform = trans.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            t_transform = trans.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])
            label_14_transform = trans.Compose([
                trans.Scale((img_size // 16, img_size // 16), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
            label_28_transform = trans.Compose([
                trans.Scale((img_size//8, img_size//8), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
            label_56_transform = trans.Compose([
                trans.Scale((img_size//4, img_size//4), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
            label_112_transform = trans.Compose([
                trans.Scale((img_size//2, img_size//2), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
            scale_size = 256
    else:
        print('----')
        '''
        transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
        ])
        '''

    if mode == 'train' or mode =='val':
        dataset = VideoData(dataset_list, data_root, alternate, len_snippet, img_transform, mode, img_size, scale_size, t_transform, label_14_transform, label_28_transform, label_56_transform, label_112_transform)
    else:
        print('----')
        #dataset = ImageData(dataset_list, data_root, transform, mode)

    return dataset

def load_list(dataset_name, data_root, mode):

    if mode == 'train':

        video_names = []
        list_num_frames = []

        video_root = os.path.join(data_root, dataset_name, 'training_set')

        for video_num in os.listdir(video_root):
            if video_num[-3:] == 'AVI':
                continue
            video_names.append(video_num)
            list_num_frames.append(len(os.listdir(os.path.join(video_root, video_num, 'images'))))


    if mode == 'val':

        video_names = []
        list_num_frames = []

        video_root = os.path.join(data_root, dataset_name, 'validation_set')

        for video_num in os.listdir(video_root):
            if video_num[-3:] == 'AVI':
                continue
            video_names.append(video_num)
            list_num_frames.append(len(os.listdir(os.path.join(video_root, video_num, 'images'))))

    return video_names, list_num_frames




class VideoData(data.Dataset):
    def __init__(self, dataset_list, data_root, alternate, len_snippet, img_transform, mode, img_size=None, scale=None, t_transform=None, label_14_transform=None, label_28_transform=None, label_56_transform=None, label_112_transform=None ):
        
        self.img_transform = img_transform
        self.mode = mode
        self.len_snippet = len_snippet
        self.alternate = alternate
        self.dataset_list = dataset_list
        self.data_root = data_root
        self.label_14_transform = label_14_transform
        self.label_28_transform = label_28_transform
        self.label_56_transform = label_56_transform
        self.label_112_transform = label_112_transform
        self.t_transform = t_transform

        if self.mode == 'train' or self.mode == 'val':
            self.video_names, self.list_num_frames = load_list(dataset_list, data_root, mode)

    def __len__(self):
        return len(self.list_num_frames)

    def __getitem__(self, index):
        if self.mode == 'train':
            file_name = self.video_names[index]
            start_idx = np.random.randint(0, self.list_num_frames[index] - self.alternate * self.len_snippet + 1)
            path_clip = os.path.join(self.data_root, self.dataset_list, 'training_set', file_name, 'images')
            path_annt = os.path.join(self.data_root, self.dataset_list, 'training_set', file_name, 'maps')
        elif self.mode == 'val':
            file_name = self.video_names[index]
            start_idx = np.random.randint(0, self.list_num_frames[index] - self.alternate * self.len_snippet + 1)
            path_clip = os.path.join(self.data_root, self.dataset_list, 'validation_set', file_name, 'images')
            path_annt = os.path.join(self.data_root, self.dataset_list, 'validation_set', file_name, 'maps')

        clip_img =[]

        for i in range(self.len_snippet):

            img = Image.open(os.path.join(path_clip, '%04d.png'%(start_idx+self.alternate*i+1))).convert('RGB')
            sz = img.size

            if self.mode != 'test' and i == (self.len_snippet-1):

                clip_gt = Image.open(os.path.join(path_annt, '%04d.png'%(start_idx+self.alternate*i+1))).convert('L')
                

            clip_img.append(self.img_transform(img))
        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
        label_14 = self.label_14_transform(clip_gt)
        label_28 = self.label_28_transform(clip_gt)
        label_56 = self.label_56_transform(clip_gt)
        label_112 = self.label_112_transform(clip_gt)
        label_224 = self.t_transform(clip_gt)

        return clip_img, label_224, label_14, label_28, label_56, label_112