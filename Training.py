import torch
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from dataset_4 import *
from torch.autograd import Variable
from TempVST import TempVST
import wandb 
import math
from utils import *
import sys

def main(local_rank, num_gpus, args):

    cudnn.benchmark = True
    #torch.cuda.set_device(local_rank)
    
    net = TempVST(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)
    net.train()
    net.cuda()

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    base_params = [
        params for name, params in net.named_parameters() if ("backbone" in name)
    ]
    other_params = [
        params for name, params in net.named_parameters() if ("backbone" not in name)
    ]

    optimizer = optim.AdamW(
        [
            {"params": base_params, "lr": args.lr * 0.1},
            {"params": other_params, "lr": args.lr},
        ],
        weight_decay=1e-3
    )
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65,130,200], gamma=0.1)
    
    train_dataset = get_loader(
        args.trainset, args.data_root, args.alternate, args.len_snippet, args.img_size, mode='train'
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_gpus,
        rank=local_rank,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    
    val_dataset = get_loader(
        args.trainset, args.data_root, args.alternate, args.len_snippet, args.img_size, mode='val'
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=num_gpus,
        rank=local_rank,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True,
    )
    
    save_best_comb_loss = SaveBestModel()
    save_best_cc = SaveBestModel()
    save_best_sim = SaveBestModel()

    print(
        """
        Starting training:
            Batch size: {}
            Learning rate: {}
            Training size: {}
        """.format(
            args.batch_size, args.lr, len(train_loader.dataset)
        )
    )

    N_train = len(train_loader)*args.batch_size

    loss_weights = [1, 0.8, 0.8, 0.5, 0.5, 0.5]
    criterion = nn.BCEWithLogitsLoss()
    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)

    total_loss = AverageMeter()
    total_cc_loss = AverageMeter()
    total_sim_loss = AverageMeter()
    val_total_loss = AverageMeter()
    val_total_cc_loss = AverageMeter()
    val_total_sim_loss = AverageMeter()
    comb_total_loss = AverageMeter()
    val_comb_total_loss = AverageMeter()
   
    for epoch in range(args.epochs):
         
        print("Starting epoch {}/{}.".format(epoch + 1, args.epochs))
        print("epoch:{0}-------lr:{1}".format(epoch + 1, args.lr))

        epoch_total_loss = 0
        epoch_loss = 0
        other_loss = 0

        net.train()
        for i, data_batch in enumerate(train_loader):
           # if (i + 1) > iter_num:
           #     break
            optimizer.zero_grad()

            (
                images,
                label_224,
                label_14,
                label_28,
                label_56,
                label_112,
            ) = data_batch

            videos, label_224= (
                images.cuda(local_rank, non_blocking=True),
                label_224.cuda(local_rank, non_blocking=True)
            )
            
            label_14, label_28, label_56, label_112 = (
                label_14.cuda(),
                label_28.cuda(),
                label_56.cuda(),
                label_112.cuda(),
            )

            label_224 = label_224[:,-1,:,:,:]
            label_112 = label_112[:,-1,:,:,:]
            label_56 = label_56[:,-1,:,:,:]
            label_28 = label_28[:,-1,:,:,:]
            label_14 = label_14[:,-1,:,:,:]

            outputs_saliency = net(videos)
        
            mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency

            mask_1_16 = ((mask_1_16 - mask_1_16.min()) * (1/(mask_1_16.max() - mask_1_16.min()) * 1))
            mask_1_8 = ((mask_1_8 - mask_1_8.min()) * (1/(mask_1_8.max() - mask_1_8.min()) * 1))
            mask_1_4 = ((mask_1_4 - mask_1_4.min()) * (1/(mask_1_4.max() - mask_1_4.min()) * 1))
            mask_1_1 = ((mask_1_1 - mask_1_1.min()) * (1/(mask_1_1.max() - mask_1_1.min()) * 1))

            # benchmark loss
            loss = loss_func(mask_1_1, label_224, args)
            cc_loss = cc(mask_1_1, label_224)
            sim_loss = similarity(mask_1_1, label_224)

            comb_loss = loss - 0.5*cc_loss.cpu().data - 0.5*sim_loss.cpu().data 

            comb_total_loss.update(comb_loss.item())
            total_loss.update(loss.item())
            total_cc_loss.update(cc_loss.item())
            total_sim_loss.update(sim_loss.item())
            # saliency loss
            loss5 = loss_func(mask_1_16, label_14, args)
            loss4 = loss_func(mask_1_8, label_28, args)
            loss3 = loss_func(mask_1_4, label_56, args)

            hyper_sup_loss = (
                loss_weights[0] * loss
                + loss_weights[2] * loss3
                + loss_weights[3] * loss4
                + loss_weights[4] * loss5
            )

            epoch_total_loss += hyper_sup_loss.cpu().data.item()
            epoch_loss += loss.cpu().data.item()
            other_loss += loss.cpu().data.item()
            
            loss3.cpu().data
            loss4.cpu().data
            loss5.cpu().data
            comb_loss.cpu().data
            

            print(
                "whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- saliency loss: {3:.6f} --- kl_loss: {4:.5f} ".format(
                    (whole_iter_num + 1),
                    (i + 1) * args.batch_size / N_train,
                    hyper_sup_loss.item(),
                    loss.item(),
                    loss.item()
                )
            )

            #loss.backward()
            #hyper_sup_loss.backward()
            comb_loss.backward() 

            optimizer.step()
            

            whole_iter_num += 1
        #scheduler.step()
        if epoch % 5 == 0:
            net.eval()
            with torch.no_grad():
                for i, data_batch in enumerate(val_loader):

                    (
                        images,
                        label_224,
                        label_14,
                        label_28,
                        label_56,
                        label_112,
                    ) = data_batch

                    videos, label_224= (
                        images.cuda(local_rank, non_blocking=True),
                        label_224.cuda(local_rank, non_blocking=True)
                    )
                    
                    label_14, label_28, label_56, label_112 = (
                        label_14.cuda(),
                        label_28.cuda(),
                        label_56.cuda(),
                        label_112.cuda(),
                    )

                    label_224 = label_224[:,-1,:,:,:]
                    label_112 = label_112[:,-1,:,:,:]
                    label_56 = label_56[:,-1,:,:,:]
                    label_28 = label_28[:,-1,:,:,:]
                    label_14 = label_14[:,-1,:,:,:]

                    outputs_saliency = net(videos)
                
                    mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
                    mask_1_8 = ((mask_1_8 - mask_1_8.min()) * (1/(mask_1_8.max() - mask_1_8.min()) * 1))
                    mask_1_4 = ((mask_1_4 - mask_1_4.min()) * (1/(mask_1_4.max() - mask_1_4.min()) * 1))
                    mask_1_1 = ((mask_1_1 - mask_1_1.min()) * (1/(mask_1_1.max() - mask_1_1.min()) * 1))

                    # benchmark loss
                    loss= loss_func(mask_1_1, label_224, args)
                    cc_loss= cc(mask_1_1, label_224)
                    sim_loss = similarity(mask_1_1, label_224)

                    val_comb_loss = loss - 0.5*cc_loss.cpu().data - 0.5*sim_loss.cpu().data 

                    val_comb_total_loss.update(val_comb_loss.item())
                    val_total_loss.update(loss.item())
                    val_total_cc_loss.update(cc_loss.item())
                    val_total_sim_loss.update(sim_loss.item())
                    # saliency loss
                    
            wandb.log({ 'val_bench_total': val_total_loss.avg, 
            'val_bench_cc': val_total_cc_loss.avg, 
            'val_bench_sim': val_total_sim_loss.avg, 
            'val comb total loss': val_comb_total_loss.avg})
                 

        print("Epoch finished ! Loss: {}".format(epoch_total_loss / iter_num))
        print("Epoch finished ! Loss: {}".format(other_loss / iter_num))

        print('[{:2d}, val] avg_loss : {:.5f} cc_loss : {:.5f} sim_loss : {:.5f}'.format(epoch, total_loss.avg, total_cc_loss.avg, total_sim_loss.avg))
        sys.stdout.flush()

        save_best_cc(-val_total_cc_loss.avg, epoch, net, optimizer, 'cc','best_cc_model.pth')
        save_best_sim(-val_total_sim_loss.avg, epoch, net, optimizer, 'cc','best_sim_model.pth')
        save_best_comb_loss(val_comb_total_loss.avg, epoch, net, optimizer, 'cc','best_comb_model.pth')

        wandb.log({'epoch loss': epoch_total_loss/iter_num,
            'sal loss': epoch_loss / iter_num,
            'other_loss': other_loss/iter_num, 
            'bench_total': total_loss.avg, 
            'bench_cc': total_cc_loss.avg, 
            'bench_sim': total_sim_loss.avg, 
            'comb total loss':comb_total_loss.avg})
        #'''
        total_loss.reset()
        total_cc_loss.reset()
        total_sim_loss.reset()
        comb_total_loss.reset()

        val_total_loss.reset()
        val_total_cc_loss.reset()
        val_total_sim_loss.reset()
        val_comb_total_loss.reset()

def train_net(num_gpus, args):

    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))