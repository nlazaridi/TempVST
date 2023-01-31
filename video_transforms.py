import numbers
import random
import numpy as np
import PIL
#import skimage.transform
import torchvision
import math
import torch
from torchvision import transforms


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    elif isinstance(clip[0], torch.Tensor):
        cropped = [img[:, min_h:min_h + h, min_w:min_w + w] for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped

class RandomResizedCrop(object):

    def __init__(self, size, scale=(0.85, 1.0)):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)

        self.scale = scale
        self.toPIL = transforms.ToPILImage()
        self.resize_224 = transforms.Resize((224,224))

    
    def __call__(self, clip):
        
        scale = random.randint(self.scale[0]*100, self.scale[1]*100)
        scale = scale/100
        print(scale)
        
        if isinstance(clip[0], np.ndarray):
            height, width, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            width, height = clip[0].size
        else:
            im_c, height, width = clip[0].shape

        h = math.floor(scale*height)
        w = math.floor(scale*width)

        i = random.randint(0, math.floor((1-scale)*height))
        j = random.randint(0, math.floor((1-scale)*width))

        imgs = crop_clip(clip, i, j, h, w)

        imgs = torch.FloatTensor(torch.stack(imgs, dim=0))

        imgs = self.resize_224(imgs)
        
        return(imgs)
'''
class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated
'''