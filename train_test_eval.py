import os
import torch
import Training
import argparse 
import wandb
import Testing

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #train 
    parser.add_argument('--Training', default=False, type=bool, help='Are we training?')
    parser.add_argument('--Testing', default=True, type=bool, help='Are we testing?')
    parser.add_argument(
        "--lr_decay_gamma", default=0.1, type=int, help="learning rate decay"
    )
    parser.add_argument("--lr", default=1e-4, type=int, help="learning rate")
    parser.add_argument("--epochs", default=1000, type=int, help="epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="batch_size")
    parser.add_argument("--num_gpu", default=1, type=int, help="batch_size")
    parser.add_argument(
        "--stepvalue1", default=30000, type=int, help="the step 1 for adjusting lr"
    )
    parser.add_argument(
        "--stepvalue2", default=45000, type=int, help="the step 2 for adjusting lr"
    )
    parser.add_argument(
        "--trainset", default='DHF1K', type=str, help="Trainging set"
    )
    parser.add_argument('--data_root', default='/data2/nlazaridis/sal_dataset/', type=str, help="data path")
    parser.add_argument("--img_size", default=224, type=int, help="network input size")
    parser.add_argument('--alternate',default=2, type=int)
    parser.add_argument('--len_snippet', default=8, type=int)
    parser.add_argument(
        "--pretrained_model",
        default="80.7_T2T_ViT_t_14.pth.tar",
        type=str,
        help="load Pretrained model",
    )

    parser.add_argument('--kldiv',default=True, type=bool)
    parser.add_argument('--cc',default=False, type=bool)
    parser.add_argument('--nss',default=False, type=bool)
    parser.add_argument('--sim',default=False, type=bool)
    parser.add_argument('--nss_emlnet',default=False, type=bool)
    parser.add_argument('--nss_norm',default=False, type=bool)
    parser.add_argument('--l1',default=False, type=bool)

    parser.add_argument('--kldiv_coeff',default=1.0, type=float)
    parser.add_argument('--cc_coeff',default=-1.0, type=float)
    parser.add_argument('--sim_coeff',default=-1.0, type=float)
    parser.add_argument('--nss_coeff',default=1.0, type=float)
    parser.add_argument('--nss_emlnet_coeff',default=1.0, type=float)
    parser.add_argument('--nss_norm_coeff',default=1.0, type=float)
    parser.add_argument('--l1_coeff',default=1.0, type=float)
    args = parser.parse_args()

    # define the gpus we are going to use 
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    num_gpus = torch.cuda.device_count()

    if args.Training:
        with wandb.init(project='TempVST'):
            Training.main(0, num_gpus=num_gpus, args=args)
            Training.train_net(num_gpus=num_gpus, args=args)
    if args.Testing:
        Testing.main(0, num_gpus=num_gpus, args=args)