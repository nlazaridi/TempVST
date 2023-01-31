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
import pytorchvideo
from pytorchvideo.transforms import AugMix
import video_transforms
from torchvision.transforms import Resize

class Compose(object):
    """Composes several transforms
    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms, use_t):
        self.transforms = transforms
        self.use_t = use_t
    def __call__(self, clip):
        for i, t in enumerate(self.transforms):
            #print(use_t)
            if self.use_t[0] == False and i == 0:
                resize_224 = transforms.Resize((224,224))
                clip = resize_224(clip)
            if self.use_t[i]==True:
                clip = t(clip)
        return clip


def get_loader(dataset_list, data_root, alternate, len_snippet, img_size, mode='train'):

    if mode == 'train' or mode == 'val':

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

            use_t = [bool(random.getrandbits(1)), bool(random.getrandbits(1)),bool(random.getrandbits(1)), True]

            clip_transform = Compose([
                video_transforms.RandomResizedCrop(size=(224,224), scale=[0.80,1]),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ], use_t)
            gt_transform = Compose([
                video_transforms.RandomResizedCrop(size=(224,224), scale = [0.80,1]),
                transforms.GaussianBlur(kernel_size=3),    
            ], use_t)
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
        dataset = VideoData(dataset_list, data_root, alternate, len_snippet, img_transform, mode, img_size, scale_size, t_transform, label_14_transform, label_28_transform, label_56_transform, label_112_transform, clip_transform, gt_transform)
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
            elif drop==False and i==self.len_snippet:
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

training = True
batch_size = 1
num_gpus = 1
trainset = 'DHF1K'
data_root = '/data2/nlazaridis/sal_dataset/'
img_size = 224
alternate = np.random.randint(3)+1
len_snippet = 10

train_dataset = get_loader(
    trainset, data_root, alternate, len_snippet, img_size, mode='train'
)

images, label_224, _, _, _ , _= train_dataset[0]

f, axarr = plt.subplots(2,5) 
axarr[0,0].imshow(images[0].permute(1,2,0))
axarr[0,1].imshow(images[1].permute(1,2,0))
axarr[0,2].imshow(images[2].permute(1,2,0))
axarr[0,3].imshow(images[3].permute(1,2,0))
axarr[0,4].imshow(images[4].permute(1,2,0))
axarr[1,0].imshow(label_224[0].squeeze())
axarr[1,1].imshow(label_224[1].squeeze())
axarr[1,2].imshow(label_224[2].squeeze())
axarr[1,3].imshow(label_224[3].squeeze())
axarr[1,4].imshow(label_224[4].squeeze())

seed = np.random.randint(2147483647) # make a seed with numpy generator 
random.seed(seed) # apply this seed to img tranfsorms
torch.manual_seed(seed) # needed for torchvision 0.7
'''
new_t = AugMix()
trans_images = new_t(images)
trans_labels = new_t(label_224)
f, axarr = plt.subplots(2,5) 
axarr[0,0].imshow(trans_labels[0].squeeze())
axarr[0,1].imshow(trans_labels[1].squeeze())
axarr[0,2].imshow(trans_labels[2].squeeze())
axarr[0,3].imshow(trans_labels[3].squeeze())
axarr[0,4].imshow(trans_labels[4].squeeze())
axarr[1,0].imshow(label_224[0].squeeze())
axarr[1,1].imshow(label_224[1].squeeze())
axarr[1,2].imshow(label_224[2].squeeze())
axarr[1,3].imshow(label_224[3].squeeze())
axarr[1,4].imshow(label_224[4].squeeze())

f, axarr = plt.subplots(2,5) 
axarr[0,0].imshow(images[0].permute(1,2,0))
axarr[0,1].imshow(images[1].permute(1,2,0))
axarr[0,2].imshow(images[2].permute(1,2,0))
axarr[0,3].imshow(images[3].permute(1,2,0))
axarr[0,4].imshow(images[4].permute(1,2,0))
axarr[1,0].imshow(trans_images[0].permute(1,2,0))
axarr[1,1].imshow(trans_images[1].permute(1,2,0))
axarr[1,2].imshow(trans_images[2].permute(1,2,0))
axarr[1,3].imshow(trans_images[3].permute(1,2,0))
axarr[1,4].imshow(trans_images[4].permute(1,2,0))

random.seed(seed) # apply this seed to img tranfsorms
torch.manual_seed(seed) # needed for torchvision 0.7
trans2_images = new_t(images)


f, axarr = plt.subplots(2,5) 
axarr[0,0].imshow(trans2_images[0].permute(1,2,0))
axarr[0,1].imshow(trans2_images[1].permute(1,2,0))
axarr[0,2].imshow(trans2_images[2].permute(1,2,0))
axarr[0,3].imshow(trans2_images[3].permute(1,2,0))
axarr[0,4].imshow(trans2_images[4].permute(1,2,0))
axarr[1,0].imshow(trans_images[0].permute(1,2,0))
axarr[1,1].imshow(trans_images[1].permute(1,2,0))
axarr[1,2].imshow(trans_images[2].permute(1,2,0))
axarr[1,3].imshow(trans_images[3].permute(1,2,0))
axarr[1,4].imshow(trans_images[4].permute(1,2,0))
'''
use_t = [bool(random.getrandbits(1)), bool(random.getrandbits(1)),bool(random.getrandbits(1))]
#use_t = [True, True, True, True]

random.seed(seed) # apply this seed to img tranfsorms
torch.manual_seed(seed) # needed for torchvision 0.7
gt_transform = Compose([
    transforms.ToTensor(),
    video_transforms.RandomResizedCrop(size=(224,224), scale = [0.80,1]),
    transforms.GaussianBlur(kernel_size=3),    
], use_t)
imgs = [gt_transform(label_224) for _ in range(4)]

fig = plt.figure(figsize=(7,3))
rows, cols = 2,2
for j in range(0, len(imgs)):
   fig.add_subplot(rows, cols, j+1)
   plt.imshow(imgs[j][0].squeeze())
   plt.xticks([])
   plt.yticks([])
plt.show()

random.seed(seed) # apply this seed to img tranfsorms
torch.manual_seed(seed) # needed for torchvision 0.7
clip_transform = Compose([
    transforms.ToTensor(),
    video_transforms.RandomResizedCrop(size=(224,224), scale=[0.80,1]),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
], use_t)




imgs = [clip_transform(images) for _ in range(5)]

fig = plt.figure(figsize=(7,3))
rows, cols = 2,3
for j in range(0, len(imgs)):
   fig.add_subplot(rows, cols, j+1)
   plt.imshow(imgs[j][0].permute(1,2,0))
   plt.xticks([])
   plt.yticks([])

fig.add_subplot(rows, cols, 6)
plt.imshow(images[0].permute(1,2,0))
plt.show()

gt_clip = gt_transform(label_224)
clip = clip_transform(images)

if use_t[0] == False:
    resize_224 = transforms.Resize((224,224))
    gt_clip = resize_224(gt_clip)
    clip = resize_224(gt_clip)

print('w8 here 3.10.8')