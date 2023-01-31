from PIL import Image
from torch.utils import data
import transforms as trans
from torchvision import transforms
import random
import os
import numpy as np
#import cv2
import torch 
import matplotlib.pyplot as plt
import video_transforms
from torchvision.transforms import Resize, ToPILImage, ToTensor
import vidaug.augmentors as va



class Compose(object):
    """Composes several transforms
    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms
        #self.make_new_use_t = True
    def __call__(self, clip):
        #if self.make_new_use_t == True:
        #    self.seed = np.random.randint(2147483647) # make a seed with numpy generator 
        #    self.make_new_use_t = False
        #else:
        #    self.make_new_use_t = True

        #random.seed(self.seed) # apply this seed to img tranfsorms
        #torch.manual_seed(self.seed) # needed for torchvision 0.7
        self.use_t = [bool(random.getrandbits(1)), bool(random.getrandbits(1)), bool(random.getrandbits(1)), bool(random.getrandbits(1)),True]    
        
        print(self.use_t)
        
        if self.use_t[0]:
            clip = torch.flip(clip, dims=(0,))
        
        for i, t in enumerate(self.transforms):
            #print(self.use_t)
            
            if self.use_t[1] == False and i == 0:
                resize_224 = transforms.Resize((224,224))
                clip = resize_224(clip)
            if self.use_t[i+1]==True:
                clip = t(clip)
        return clip


def get_loader(dataset_list, data_root, alternate, len_snippet, img_size, mode='train'):

    if mode == 'train':

            img_transform = trans.Compose([
                    #transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            t_transform = trans.Compose([
                #transforms.Resize((224,224)),
                transforms.ToTensor()
            ])
            
            label_14_transform = trans.Compose([
                Resize((img_size // 16, img_size // 16), interpolation=Image.NEAREST),
                #trans.Scale((img_size // 16, img_size // 16), interpolation=Image.NEAREST),
                #transforms.ToTensor(),
            ])
            label_28_transform = trans.Compose([
                Resize((img_size // 8, img_size // 8), interpolation=Image.NEAREST),
                #trans.Scale((img_size//8, img_size//8), interpolation=Image.NEAREST),
                #transforms.ToTensor(),
            ])
            label_56_transform = trans.Compose([
                Resize((img_size // 4, img_size // 4), interpolation=Image.NEAREST),
                #trans.Scale((img_size//4, img_size//4), interpolation=Image.NEAREST),
                #transforms.ToTensor(),
            ])
            label_112_transform = trans.Compose([
                Resize((img_size // 2 , img_size // 2), interpolation=Image.NEAREST),
                #trans.Scale((img_size//2, img_size//2), interpolation=Image.NEAREST),
                #transforms.ToTensor(),
            ])

            #use_t = [bool(random.getrandbits(1)), bool(random.getrandbits(1)),bool(random.getrandbits(1)), True]

            clip_transform = Compose([
                #transforms.RandomRotation((5,20)),
                video_transforms.RandomResizedCrop(size=(224,224), scale=[0.80,1]),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            gt_transform = Compose([
                #transforms.RandomRotation((5,20)),
                video_transforms.RandomResizedCrop(size=(224,224), scale = [0.80,1])            ])
            scale_size = 256

    elif mode=='val':

            img_transform = trans.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            t_transform = trans.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])
            
            label_14_transform = trans.Compose([
                Resize((img_size // 16, img_size // 16), interpolation=Image.NEAREST),
                #trans.Scale((img_size // 16, img_size // 16), interpolation=Image.NEAREST),
                #transforms.ToTensor(),
            ])
            label_28_transform = trans.Compose([
                Resize((img_size // 8, img_size // 8), interpolation=Image.NEAREST),
                #trans.Scale((img_size//8, img_size//8), interpolation=Image.NEAREST),
                #transforms.ToTensor(),
            ])
            label_56_transform = trans.Compose([
                Resize((img_size // 4, img_size // 4), interpolation=Image.NEAREST),
                #trans.Scale((img_size//4, img_size//4), interpolation=Image.NEAREST),
                #transforms.ToTensor(),
            ])
            label_112_transform = trans.Compose([
                Resize((img_size // 2 , img_size // 2), interpolation=Image.NEAREST),
                #trans.Scale((img_size//2, img_size//2), interpolation=Image.NEAREST),
                #transforms.ToTensor(),
            ])

            #use_t = [bool(random.getrandbits(1)), bool(random.getrandbits(1)),bool(random.getrandbits(1)), True]

            clip_transform = trans.Compose([
                #transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            gt_transform = trans.Compose([
                #transforms.ToTensor()  
            ])
            scale_size = 256

    elif mode=='test':

            img_transform = trans.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            t_transform = trans.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])

            scale_size=256
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
        dataset = VideoData(dataset_list, data_root, alternate, len_snippet, img_transform, mode, img_size, scale_size, t_transform, label_14_transform, label_28_transform, label_56_transform, label_112_transform, clip_transform, gt_transform)
    else:
        dataset = TestVideoData(dataset_list, data_root, alternate, len_snippet, img_transform, mode, img_size, scale_size, t_transform)
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

    if mode == 'test':

        video_names = []
        list_num_frames = []

        video_root = os.path.join(data_root, dataset_name, 'test_set')
       
        for video_num in os.listdir(video_root):
            #'''
            if video_num[-3:] == 'AVI' or video_num == 'results':
                continue
        
            for frame_num in range(len(os.listdir(os.path.join(video_root, video_num, 'images')))):
                video_names.append(video_num)
                list_num_frames.append(frame_num)
        '''
        video_num = '0701'
        for frame_num in range(len(os.listdir(os.path.join(video_root, video_num, 'images')))):
            video_names.append(video_num)
            list_num_frames.append(frame_num)
        '''
    return video_names, list_num_frames    

class TestVideoData(data.Dataset):
    def __init__(self, dataset_list, data_root, alternate, len_snippet, img_transform, mode, img_size=None, scale=None, t_transform=None):
        self.img_transform = img_transform
        self.mode = mode
        self.len_snippet = len_snippet
        self.alternate = alternate
        self.dataset_list = dataset_list
        self.data_root = data_root

        if self.mode == 'test':
            self.video_names, self.list_frame_num = load_list(dataset_list, data_root, mode)

    def __len__(self):
        return len(self.list_frame_num)
    
    def __getitem__(self, index):

        if self.mode == 'test':
            file_name = self.video_names[index]
            start_idx = self.list_frame_num[index]
            path_clip = os.path.join(self.data_root, self.dataset_list, 'test_set', file_name, 'images')

        clip_img = []
        
        for i in range (self.len_snippet):
            if start_idx < (self.len_snippet*self.alternate-1):
                img = Image.open(os.path.join(path_clip, '%04d.png'%(start_idx+self.alternate*i+1))).convert('RGB')
            else:
                img = Image.open(os.path.join(path_clip,'%04d.png'%(start_idx - (self.len_snippet-i-1)*self.alternate))).convert('RGB')

            clip_img.append(self.img_transform(img))

        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
        if start_idx < (self.len_snippet*self.alternate-1):
           clip_img = torch.flip(clip_img, dims=(0,))

        return clip_img, file_name, start_idx

class VideoData(data.Dataset):
    def __init__(self, dataset_list, data_root, alternate, len_snippet, img_transform, mode, img_size=None, scale=None, t_transform=None, label_14_transform=None, label_28_transform=None, label_56_transform=None, label_112_transform=None, clip_transform=None, gt_transform=None ):
        
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
        self.clip_transform = clip_transform
        self.gt_transform = gt_transform

        if self.mode == 'train' or self.mode == 'val':
            self.video_names, self.list_num_frames = load_list(dataset_list, data_root, mode)

    def __len__(self):
        return len(self.list_num_frames)

    def __getitem__(self, index):
        
        if self.mode == 'train':
            file_name = self.video_names[index]
            start_idx = np.random.randint(0, self.list_num_frames[index] - self.alternate * (self.len_snippet+1) + 1)
            path_clip = os.path.join(self.data_root, self.dataset_list, 'training_set', file_name, 'images')
            path_annt = os.path.join(self.data_root, self.dataset_list, 'training_set', file_name, 'maps')
        elif self.mode == 'val':
            file_name = self.video_names[index]
            start_idx = np.random.randint(0, self.list_num_frames[index] - self.alternate * self.len_snippet + 1)
            path_clip = os.path.join(self.data_root, self.dataset_list, 'validation_set', file_name, 'images')
            path_annt = os.path.join(self.data_root, self.dataset_list, 'validation_set', file_name, 'maps')

        clip_img = []
        #clip_gt = []
        label_14 = []
        label_28 = []
        label_56 = []
        label_112 = []
        label_224 = []

        drop = False
        if bool(random.getrandbits(1)) and bool(random.getrandbits(1)):
            drop = True
            drop_frame = np.random.randint(self.len_snippet)
        
            

        for i in range(self.len_snippet+1):
            if drop and i == drop_frame and self.mode=='train':
                continue
            elif drop==False and i==self.len_snippet and self.mode=='train':
                continue
            elif i==self.len_snippet and self.mode=='val':
                continue

            img = Image.open(os.path.join(path_clip, '%04d.png'%(start_idx+self.alternate*i+1))).convert('RGB')
            sz = img.size

            if self.mode != 'test':
              #and i == (self.len_snippet-1):

              label = Image.open(os.path.join(path_annt, '%04d.png'%(start_idx+self.alternate*i+1))).convert('L')
                

            clip_img.append(self.img_transform(img))
            #clip_gt.append(self.t_transform(label))
            #label_14.append(self.label_14_transform(label))
            #abel_28.append(self.label_28_transform(label))
            #label_56.append(self.label_56_transform(label))
            #label_112.append(self.label_112_transform(label))
            label_224.append(self.t_transform(label))
        
        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7

        clip_img = self.clip_transform(clip_img)

        label_224 = torch.FloatTensor(torch.stack(label_224, dim=0))

        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7

        label_224 = self.gt_transform(label_224)

        label_14 = self.label_14_transform(label_224)
        label_28 = self.label_28_transform(label_224)
        label_56 = self.label_56_transform(label_224)
        label_112 = self.label_112_transform(label_224)
        #clip_gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))
        #label_14 = torch.FloatTensor(torch.stack(label_14, dim=0))
        #label_28 = torch.FloatTensor(torch.stack(label_28, dim=0))
        #label_56 = torch.FloatTensor(torch.stack(label_56, dim=0))
        #label_112 = torch.FloatTensor(torch.stack(label_112, dim=0))
        

        return clip_img, label_224, label_14, label_28, label_56, label_112