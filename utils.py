import torch
import torch.nn as nn
from loss import *
import cv2
from torchvision import transforms, utils
from PIL import Image


def get_loss(pred_map, gt, args):
    loss = torch.FloatTensor([0.0]).to('cuda')
    #loss = torch.FloatTensor([0.0])
    if args.kldiv:
        #pred_map = pred_map.to('cpu')
        #gt = gt.to('cpu')
        loss += args.kldiv_coeff * kldiv(pred_map, gt)
    if args.cc:
        loss += args.cc_coeff * cc(pred_map, gt)
    if args.l1:
        loss += args.l1_coeff * criterion(pred_map, gt)
    if args.sim:
        loss += args.sim_coeff * similarity(pred_map, gt)

    return loss

def loss_func(pred_map, gt, args):
    #loss = torch.FloatTensor([0.0]).cuda()
    loss = torch.FloatTensor([0.0])
    criterion = nn.L1Loss()
    if len(gt.size()) == 2:
        gt = gt.unsqueeze(0)
    assert pred_map.size() == gt.size()

    if len(pred_map.size()) == 4:
        ''' Clips: BxClXHxW '''
        assert pred_map.size(0)==args.batch_size
        #pred_map = pred_map.permute((1,0,2,3))
        #gt = gt.permute((1,0,2,3))

        for i in range(pred_map.size(0)):
            loss += (get_loss(pred_map[i], gt[i], args)).to('cpu')

        loss /= pred_map.size(0)
        return loss
    
    return get_loss(pred_map, gt, args)

class AverageMeter(object):

    '''Computers and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img,(k_size,k_size),0)
    return torch.FloatTensor(bl)

def img_save(tensor, fp, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    grid = utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    ndarr = torch.round(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)).to('cpu', torch.uint8).numpy()
    ndarr = ndarr[:,:,0]
    im = Image.fromarray(ndarr)
    exten = fp.split('.')[-1]
    if exten=="png":
        im.save(fp, format=format)
    else:
        im.save(fp, format=format, quality=100) #for jpg


def num_params(model):
    return sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())


class SaveBestModel:

    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion, path_file
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path_file)