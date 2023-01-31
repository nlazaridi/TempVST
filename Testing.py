import torch
from TempVST import TempVST
import wandb 
import math
from utils import *
import sys
from dataset_4 import *
from torchvision.utils import save_image
from tqdm import tqdm

def main(local_rank, num_gpus, args):
    #load model with best cc score 
    best_cc_checkpoint = torch.load('best_cc_model.pth')
    net = TempVST(args)
    net.load_state_dict(best_cc_checkpoint['model_state_dict'])

    #pass model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)
    net.eval()
    net.cuda()

    #init test set
    test_dataset = get_loader(
        args.trainset, args.data_root, args.alternate, args.len_snippet, args.img_size, mode='test'
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=num_gpus,
        rank=local_rank,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=test_sampler,
        drop_last=True,
    )
    
    init_size = transforms.Resize((360,640))

    #inderence 
    with torch.no_grad():
        for i, data_batch in enumerate(tqdm(test_loader)):
            # load video samples
            (
                frames,
                file_name,
                start_idx 
            ) = data_batch
            video = frames.cuda(local_rank, non_blocking=True)
            #forward pass
            mask = net(video)[3]
            #save mask
            mask = ((mask - mask.min()) * (1/(mask.max() - mask.min()) * 1))
            mask = init_size(mask)
            folder_path = os.path.join(args.data_root,'DHF1K/test_set/results', file_name[0])
            if os.path.exists(folder_path) != True:
                os.mkdir(folder_path)
            img_path = os.path.join(folder_path, '%04d.png'%(start_idx+1))    
            save_image(mask,img_path)
            #print(img_path)  