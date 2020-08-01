"""
Script for training DCSNet.
"""

import os, sys
import logging
import random
import shutil
import time
import sys
import argparse
import numpy as np
import sigpy.plot as pl
import torch
import sigpy as sp
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
from unet.unet_model import UNet3
from unet.vgg import Vgg16
matplotlib.use('TkAgg')
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils.datasets import SliceData
import networks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransform:
    """
    Data Transformer for training super resolution networks
    """

    def __init__(self, args, use_seed=False):
        """
        Args:
            mask_func (utils.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.mask_func = None
#         self.use_seed = use_seed
        self.rng = np.random.RandomState()

#         # Method for initializing network
#         self.slwin_init = args.slwin_init # sliding window
#         self.emaps = args.num_emaps
        
    def augment(self, MRF):
        """
        Apply phase augmentations.
        """
        theta = np.random.rand()
        MRF = MRF*np.exp(1j*2*np.pi*theta)
        MRF_out = np.concatenate((np.real(MRF),np.imag(MRF)))
        return MRF_out

    def __call__(self, MRF, T1,T2,FLAIR,MRF_avg):
#         Input = sp.resize(Input,(38,46,28))
#         Target = sp.resize(Target,(76,92,56))
#         im_all = np.concatenate((MRF_avg[None,...],T1[None,...],T2[None,...],FLAIR[None,...]))
#         pl.ImagePlot(im_all)
#         print(im_all.shape)
#         sys.exit()
        # Convert everything from numpy arrays to tensors
        
        MRF = self.augment(MRF)
#         pl.ImagePlot(MRF)
#         print(MRF.shape)        
        MRF = torch.tensor(MRF,dtype=torch.float32)
        T1 = torch.tensor(T1[None,...],dtype=torch.float32)
        T2 = torch.tensor(T2[None,...],dtype=torch.float32)
        FLAIR = torch.tensor(FLAIR[None,...],dtype=torch.float32)
        MRF_avg = torch.tensor(MRF_avg[None,...],dtype=torch.float32)

        return MRF, T1,T2,FLAIR,MRF_avg


def create_datasets(args):
    train_data = SliceData(
        root=os.path.join(str(args.data_path), 'Train'),
        transform=DataTransform(args),
        sample_rate=args.sample_rate
    )
    dev_data = SliceData(
        root=os.path.join(str(args.data_path), 'Validate'),
        transform=DataTransform(args, use_seed=True),
        sample_rate=args.sample_rate
    )
    return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
#     print(train_data[2])
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.num_workers,
        num_workers=16,
        pin_memory=True,
    )
#     sys.exit()
    return train_loader, dev_loader


def compute_metrics(args, model,vgg, MRF,T1,T2,FLAIR,MRF_avg):
    # Load input, sensitivity maps, and target images onto device
    # Forward pass through network
    T1_out,T2_out,FLAIR_out = model(MRF)
#     pl.ImagePlot(output.detach().cpu().numpy())
#     sys.exit()
#     print(maps.shape)
#     pl.ImagePlot(output.detach().cpu())
#     sys.exit()
    # Undo normalization from pre-processing
#     sys.exit()
    if args.loss_normalized == True:
        s1 = ((torch.sum(T1_out*T1,(1,2,3))/torch.sum(T1_out**2,(1,2,3))).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).detach()
        s2 = ((torch.sum(T2_out*T2,(1,2,3))/torch.sum(T2_out**2,(1,2,3))).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).detach()
        s3 = ((torch.sum(FLAIR_out*FLAIR,(1,2,3))/torch.sum(FLAIR_out**2,(1,2,3))).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).detach()
        T1_out = T1_out*s1
        T2_out = T2_out*s2
        FLAIR_out = FLAIR_out*s3
        
#         print(args.loss_uflossdir)
#     sys.exit()
    
    # Compute image quality metrics from complex-valued images
#     cplx_error = abs(output - target)
    L1_crt = nn.L1Loss()
    loss_vgg = None
    cplx_l1 = L1_crt(T1,T1_out) + L1_crt(T2,T2_out) + L1_crt(FLAIR,FLAIR_out)
    if vgg is not None:
        T1_out_img = T1_out.repeat(1,3,1,1)
        T1_out_img = T1_out_img/T1_out_img.max()
        T1_img = T1.repeat(1,3,1,1)
        T1_img = T1_img/T1_img.max()
        T1_out_f = vgg(T1_out_img).relu2_2
        T1_f = vgg(T1_img).relu2_2
        
        T2_out_img = T2_out.repeat(1,3,1,1)
        T2_out_img = T2_out_img/T2_out_img.max()
        T2_img = T2.repeat(1,3,1,1)
        T2_img = T2_img/T2_img.max() 
        T2_out_f = vgg(T2_out_img).relu2_2
        T2_f = vgg(T2_img).relu2_2        
        
        FLAIR_out_img = FLAIR_out.repeat(1,3,1,1)
        FLAIR_out_img = FLAIR_out_img/FLAIR_out_img.max()
        FLAIR_img = FLAIR.repeat(1,3,1,1)
        FLAIR_img = FLAIR_img/FLAIR_img.max()     
        FLAIR_out_f = vgg(FLAIR_out_img).relu2_2
        FLAIR_f = vgg(FLAIR_img).relu2_2         
        
        loss_vgg = L1_crt(T1_out_f,T1_f) + L1_crt(T2_out_f,T2_f) + L1_crt(FLAIR_out_f,FLAIR_f)
    # Compute image quality metrics from magnitude images
#     mag_error = torch.abs(cplx.abs(output) - cplx.abs(target))
#     mag_l1 = torch.mean(mag_error)
#     mag_l2 = torch.sqrt(torch.mean(mag_error**2))
#     mag_psnr = 20 * torch.log10(scale / mag_l2)
    
    
    
    return cplx_l1,loss_vgg,T1_out,T2_out,FLAIR_out  


def train_epoch(args, epoch, netG, netD, data_loader, optimizer_G,optimizer_D, writer):
    netG.train()
    if args.adv:
        netD.train()
        criterionGAN = networks.GANLoss("vanilla").to(args.device)
#     avg_l2 = None
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    print(len(data_loader))
    if args.vgg_perceptual:
        vgg = Vgg16(requires_grad=False).to(args.device)
    else:
        vgg = None
    for iter, data in enumerate(data_loader):
        MRF, T1,T2,FLAIR,MRF_avg = data
        MRF = MRF.to(args.device)
        T1 = T1.to(args.device)
        T2 = T2.to(args.device)
        FLAIR = FLAIR.to(args.device)
        MRF_avg = MRF_avg.to(args.device)
        
        # Compute image quality metrics
        l1_loss, vgg_loss,T1_out,T2_out,FLAIR_out = compute_metrics(args, netG,vgg, MRF,T1,T2,FLAIR,MRF_avg)
#         if iter == 50:
#             pl.ImagePlot(T2_out.detach().cpu().numpy())
#         print(l1_loss,vgg_loss)
        #Update D
        if args.adv:
            for param in netD.parameters():
                param.requires_grad = True
            optimizer_D.zero_grad()
            if args.conditional:
                fake_AB = torch.cat((MRF_avg,T1_out,T2_out,FLAIR_out),1)
            else:
                fake_AB = torch.cat((T1_out,T2_out,FLAIR_out),1)
                
            pred_fake = netD(fake_AB.detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            if args.conditional:
                real_AB = torch.cat((MRF_avg,T1,T2,FLAIR),1)
            else:
                real_AB = torch.cat((T1,T2,FLAIR),1)
            pred_real = netD(real_AB)
            loss_D_real = criterionGAN(pred_real, True)
            loss_D = (loss_D_fake+loss_D_real)*0.5
            loss_D.backward()
            optimizer_D.step()  
            for param in netD.parameters():
                param.requires_grad = False
                
            #Update G
            optimizer_G.zero_grad()
            if args.conditional:
                fake_AB = torch.cat((MRF_avg,T1_out,T2_out,FLAIR_out),1)
            else:
                fake_AB = torch.cat((T1_out,T2_out,FLAIR_out),1)
            pred_fake = netD(fake_AB)
            loss_G_GAN = criterionGAN(pred_fake, True)
            if vgg_loss is not None:
                loss_G = l1_loss + vgg_loss*args.vgg_weight + loss_G_GAN*args.gan_weight
            else:
                loss_G = l1_loss + loss_G_GAN*args.gan_weight
            loss_G.backward()
            optimizer_G.step()
        
        else:
            if vgg_loss is not None:
                loss_G = l1_loss + vgg_loss*args.vgg_weight
            else:
                loss_G = l1_loss
            loss_G.backward()
            optimizer_G.step()
        

        avg_loss_G = 0.99 * avg_loss_G + 0.01 * loss_G.item() if iter > 0 else loss_G.item()
        if args.adv:
            avg_loss_D = 0.99 * avg_loss_D + 0.01 * loss_D.item() if iter > 0 else loss_D.item()
        else:
            avg_loss_D = None
#         if ufloss is not None:
#             avg_l2 = 0.99 * avg_l2 + 0.01 * l2_loss.item() if iter > 0 else l2_loss.item()

#             avg_ufloss = 0.99 * avg_ufloss + 0.01 * ufloss.item() if iter > 0 else ufloss.item()
        

        # Write out summary
        writer.add_scalar('Train_Loss_G', loss_G.item(), global_step + iter)
        if args.adv:
            writer.add_scalar('Train_Loss_D', loss_D.item(), global_step + iter)

#         writer.add_scalar('Train_mPSNR', mpsnr.item(), global_step + iter)
#         if ufloss is not None:
#             writer.add_scalar('Train_L2loss', l2_loss.item(), global_step + iter)
#             writer.add_scalar('Train_UFLoss', ufloss.item(), global_step + iter)
        
        
#         print(loss.item(),l2_loss.item())
        if iter % args.report_interval == 0:
            if args.adv:
                logging.info(
                    f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                    f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                    f'Loss_G = {loss_G.item():.4g} Avg G Loss = {avg_loss_G:.4g}'
                    f'Loss_D = {loss_D.item():.4g} Avg D Loss = {avg_loss_D:.4g}'
                    f'Time = {time.perf_counter() - start_iter:.4f}s',
                )
            else:
                                logging.info(
                    f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                    f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                    f'Loss_G = {loss_G.item():.4g} Avg Loss = {avg_loss_G:.4g}'
                    f'Time = {time.perf_counter() - start_iter:.4f}s',
                )
            # Write images into summary
#             visualize(args, global_step + iter, model, data_loader, writer)

        start_iter = time.perf_counter()

    return avg_loss_G,avg_loss_D,time.perf_counter() - start_epoch


def evaluate(args, epoch, netG, data_loader, writer):
    netG.eval()
    losses = []
    if args.vgg_perceptual:
        vgg = Vgg16(requires_grad=False).to(args.device)
    else:
        vgg = None
#     mpsnr_vals = []
#     l2_vals = []
#     ufloss_vals = []
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            MRF, T1,T2,FLAIR,MRF_avg = data
            MRF = MRF.to(args.device)
            T1 = T1.to(args.device)
            T2 = T2.to(args.device)
            FLAIR = FLAIR.to(args.device)
            MRF_avg = MRF_avg.to(args.device)

            # Compute image quality metrics
            l1_loss, vgg_loss,_,_,_ = compute_metrics(args, netG,vgg, MRF,T1,T2,FLAIR,MRF_avg)
            loss_G = l1_loss+args.vgg_weight*vgg_loss
            
            losses.append(loss_G.item())
            

        writer.add_scalar('Val_Loss', np.mean(losses), epoch)
#         writer.add_scalar('Val_mPSNR', np.mean(mpsnr_vals), epoch)
#         if ufloss is not None:
#             writer.add_scalar('Val_L2loss', np.mean(l2_vals), epoch)
#             writer.add_scalar('Val_UFLoss', np.mean(ufloss_vals), epoch)
    return np.mean(losses),time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer, is_training=True):
    def save_image(image, tag):
        image = image.permute(0,3,1,2)
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=1, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            # Load all data arrays
            input, maps, init, target, mean, std, norm = data
            input = input.to(args.device)
            maps = maps.to(args.device)
            init = init.to(args.device)
            target = target.to(args.device)

            # Data dimensions (for my own reference)
            #  image size:  [batch_size, nx,   ny, nt, nmaps, 2]
            #  kspace size: [batch_size, nkx, nky, nt, ncoils, 2]
            #  maps size:   [batch_size, nkx,  ny,  1, ncoils, nmaps, 2]

            # Initialize signal model
            A = T.SenseModel(maps)

            # Compute DL recon
            output = model(input, maps, init_image=init)

            # Slice images
            init = init[:,:,:,10,0,None]
            output = output[:,:,:,10,0,None]
            target = target[:,:,:,10,0,None]
            mask = cplx.get_mask(input[:,-1,:,:,0,:]) # [b, y, t, 2]

            # Save images to summary
            tag = 'Train' if is_training else 'Val'
            all_images = torch.cat((init, output, target), dim=2)
            save_image(cplx.abs(all_images), '%s_Images' % tag)
            save_image(cplx.angle(all_images), '%s_Phase' % tag)
            save_image(cplx.abs(output - target), '%s_Error' % tag)
            save_image(mask.permute(0,2,1,3), '%s_Mask' % tag)

            break


def save_model(args, exp_dir, epoch, netG,netD, optimizer_G,optimizer_D, best_dev_loss, is_new_best):
    if args.adv:
        torch.save(
            {
                'epoch': epoch,
                'args': args,
                'netG': netG.state_dict(),
                'netD': netD.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'best_dev_loss': best_dev_loss,
                'exp_dir': exp_dir
            },
            f=os.path.join(exp_dir, 'model_epoch%d.pt'%(epoch))
        )
    else:
        torch.save(
            {
                'epoch': epoch,
                'args': args,
                'netG': netG.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'best_dev_loss': best_dev_loss,
                'exp_dir': exp_dir
            },
            f=os.path.join(exp_dir, 'model_epoch%d.pt'%(epoch))
        )        
    if is_new_best:
        shutil.copyfile(os.path.join(exp_dir, 'model_epoch%d.pt'%(epoch)), 
                        os.path.join(exp_dir, 'best_model.pt'))


def build_generator(args):
    net_generator = UNet3(n_channels=1000, n_classes=1).to(args.device)
#     __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
#                  num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
#     if args.modl_flag == True:
#         print("Using MoDL for training")
#         model = UnrolledModelM(args).to(args.device)
#     else:
#         print("Using DL-ESPIRiT for training")
#         model = UnrolledModel(args).to(args.device)
    return net_generator


def build_split_model(args):
    # Do not use yet - work in progress
    num_gpus = torch.cuda.device_count()
    models = [None] * num_gpus
    for i in range(num_gpus):
        device_name = torch.cuda.get_device_name(i)
        print('Initializing network %d on GPU%d (%s)' % (i, i, device_name))
        models[i] = UnrolledModel(args_device).to('cuda:%d' % i)
    return nn.ModuleList(models)


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    if "num_workers" not in args:
        args.num_workers = 16
    netG = build_generator(args)
    netD = None
    if args.adv:
        print("Using adversarial loss")
        if args.conditional:
            netD = networks.define_D(4,64,'basic').to(args.device)
        else:
            netD = networks.define_D(3,64,'basic').to(args.device)
    if args.data_parallel:
        netG = torch.nn.DataParallel(netG) 
        if args.adv:
            netD = torch.nn.DataParallel(netD) 
    netG.load_state_dict(checkpoint['netG'])
    if args.adv:
        netD.load_state_dict(checkpoint['netD'])
    
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.adv:
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer_D = None
    
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    if args.adv:
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    return checkpoint, netG,netD, optimizer_G,optimizer_D


def build_optim(args, netG,netD):
    optimizer = list([])
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.adv:
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.append(optimizer_G)
    if args.adv:
        optimizer.append(optimizer_D)
    return optimizer


def main(args):
    # Create model directory if it doesn't exist
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    writer = SummaryWriter(log_dir=args.exp_dir)

    if int(args.device_num) > -1:
        logger.info(f'Using GPU device {args.device_num}...')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num
        args.device = 'cuda'
    else:
        logger.info('Using CPU...')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        args.device = 'cpu'

    if args.resume:
        checkpoint, netG,netD, optimizer_G,optimizer_D = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        netG = build_generator(args)
        netD = None
        if args.adv:
            print("Using adversarial loss")
            if args.conditional:
                netD = networks.define_D(4,64,'basic').to(args.device)
            else:
                netD = networks.define_D(3,64,'basic').to(args.device)
        if args.data_parallel:
            netG = torch.nn.DataParallel(netG) 
            if args.adv:
                netD = torch.nn.DataParallel(netD) 
            
        optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.adv:
            optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer_D = None
        best_dev_loss = 1e9
        start_epoch = 0
    
    logging.info(args)
    logging.info(netG)

    for name, param in netG.named_parameters():
        if param.requires_grad:
            print(name)

    torch.cuda._lazy_init()

    train_loader, dev_loader = create_data_loaders(args)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, args.lr_step_size, args.lr_gamma)
    if args.adv:
        scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, args.lr_step_size, args.lr_gamma)
    
    for epoch in range(start_epoch, args.num_epochs):
        train_loss_G,train_loss_D,train_time = train_epoch(args, epoch, netG,netD, train_loader, optimizer_G,optimizer_D, writer)
        dev_loss,dev_time = evaluate(args, epoch, netG, dev_loader, writer)
        scheduler_G.step(epoch)
        if args.adv:
            scheduler_D.step(epoch)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, netG,netD, optimizer_G,optimizer_D, best_dev_loss, is_new_best)
#         if args.loss_uflossdir is not None:
#             logging.info(
#             f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} TrainL2Loss = {train_l2:.4g} TrainUFLoss = {train_ufloss:.4g}'
#             f'DevLoss = {dev_loss:.4g} DevL2Loss = {dev_l2:.4g} DevUFLoss = {dev_ufloss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
#         )
#         else:
        if args.adv:
            logging.info(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss_G = {train_loss_G:.4g} TrainLoss_D = {train_loss_D:.4g}'
                f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
            )
        else:
            logging.info(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss_G = {train_loss_G:.4g}'
                f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
            )
    writer.close()


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Training script for DCSNet.")
    # Network parameters
    
    parser.add_argument('--fix-step-size', type=bool, default=True, help='Fix unrolled step size')
    parser.add_argument('--circular-pad', type=bool, default=True, help='Flag to turn on circular padding')
    parser.add_argument('--slwin-init', action='store_true', help='If set, will use sliding window initialization.')
    parser.add_argument('--share-weights', action='store_true',default=False, help='If set, will use share weights between unrolled iterations.')

    # Data parameters
    parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--sample-rate', type=float, default=1.0, help='Fraction of total volumes to include')
    parser.add_argument('--patch-size', default=64, type=int, help='Resolution of images')
    parser.add_argument('--num-emaps', type=int, default=1, help='Number of ESPIRiT maps')

    # Undersampling parameters
    parser.add_argument('--accelerations', nargs='+', default=[10, 15], type=int,
                        help='Range of acceleration rates to simulate in training data.')

    # Training parameters
    parser.add_argument('--batch-size', default=1, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=500,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.5, # 0.1 
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--num-workers', type=int, default=16, help='number of workers')
    # Miscellaneous parameters
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--report-interval', type=int, default=3, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device-num', type=str, default='0',
                        help='Which device to train on.')
    parser.add_argument('--exp-dir', type=str, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    
    # For UFLoss
    parser.add_argument('--loss-normalized',action='store_true',default=False, help='If set, will use the normalization invariant loss')
    parser.add_argument('--loss-type', type=int, default=1, help='loss type (1 or 2)')
    parser.add_argument('--uflossfreq', type=int, default=10, help='ufloss frequency')
    
    parser.add_argument('--loss-uflossdir', type=str,default=None,help='Path to the UFLoss mapping network')
    parser.add_argument('--ufloss-weight', type=float,default=2,help='Weight of the UFLoss')
    parser.add_argument('--conv-type', type=str,default="conv3d",help='convolution type')
    parser.add_argument('--vgg-perceptual',action='store_true',default=False, help='If set, will use perceptual loss')
    parser.add_argument('--vgg-weight', type=float, default=0.05, help='weight for vgg'), # 0.1 
    parser.add_argument('--gan-weight', type=float, default=0.05, help='weight for gan'), # 0.1 
    
    parser.add_argument('--adv',action='store_true',default=False, help='If set, will use adversarial loss')
    
    parser.add_argument('--conditional',action='store_true',default=False, help='If set, will use conditional gan')
    
    
    return parser


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
