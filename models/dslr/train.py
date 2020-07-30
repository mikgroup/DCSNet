"""

"""

import os, sys
import logging
import random
import shutil
import time
import argparse
import numpy as np

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# import custom libraries
from utils import transforms as T
from utils import subsample as ss
from utils import complex_utils as cplx

# import custom classes
from utils.datasets import SliceData
from utils.subsample import VDktMaskFunc
from unrolledLLR import UnrolledModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransform:
    """
    Data Transformer for training unrolled reconstruction models.
    """

    def __init__(self, mask_func, args, use_seed=False):
        """
        Args:
            mask_func (utils.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.rng = np.random.RandomState()

        # Method for initializing network
        self.slwin_init = args.slwin_init # sliding window
        self.num_basis = args.num_basis
        self.block_size = args.block_size
        self.overlapping = args.overlapping


    def augment(self, kspace, target, seed):
        """
        Apply random data augmentations.
        """
        self.rng.seed(seed)

        # Random flips through time
        if self.rng.rand() > 0.5:
            kspace = torch.flip(kspace, dims=(3,))
            target = torch.flip(target, dims=(3,))

        return kspace, target


    def __call__(self, kspace, maps, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        seed = None if not self.use_seed else tuple(map(ord, fname))

        # Convert everything from numpy arrays to tensors
        kspace = cplx.to_tensor(kspace).unsqueeze(0)
        maps   = cplx.to_tensor(maps).unsqueeze(0)
        target = cplx.to_tensor(target).unsqueeze(0)
        norm = torch.sqrt(torch.mean(cplx.abs(target)**2))

        # Apply random data augmentation
        kspace, target = self.augment(kspace, target, seed)

        # Undersample k-space data
        masked_kspace, mask = ss.subsample(kspace, self.mask_func, seed)

        # Initialize ESPIRiT model
        A = T.SenseModel(maps)

        # Compute normalization factor (based on 95% max signal level in view-shared dataa)
        averaged_kspace = T.time_average(masked_kspace, dim=3)
        image = A(averaged_kspace, adjoint=True)
        magnitude_vals = cplx.abs(image).reshape(-1)
        k = int(round(0.05 * magnitude_vals.numel()))
        scale = torch.min(torch.topk(magnitude_vals, k).values)

        # Normalize k-space and target images
        masked_kspace /= scale
        target /= scale
        mean = torch.tensor([0.0], dtype=torch.float32)
        std = scale

        # Compute network initialization
        if self.slwin_init:
            init_image = A(T.sliding_window(masked_kspace, dim=3, window_size=5), adjoint=True)
        else:
            init_image = A(masked_kspace, adjoint=True)
        L_init, R_init = T.decompose_LR(init_image, self.num_basis, block_size=self.block_size, overlapping=self.overlapping)

        # Get rid of batch dimension
        masked_kspace = masked_kspace.squeeze(0)
        maps = maps.squeeze(0)
        L_init = L_init.squeeze(0)
        R_init = R_init.squeeze(0)
        target = target.squeeze(0)

        return masked_kspace, maps, L_init, R_init, target, mean, std, norm


def create_datasets(args):
    # Generate k-t undersampling masks
    train_mask = VDktMaskFunc(args.accelerations)
    dev_mask = VDktMaskFunc(args.accelerations) 

    train_data = SliceData(
        root=os.path.join(str(args.data_path), 'train'),
        transform=DataTransform(train_mask, args),
        sample_rate=args.sample_rate
    )
    dev_data = SliceData(
        root=os.path.join(str(args.data_path), 'validate'),
        transform=DataTransform(dev_mask, args, use_seed=True),
        sample_rate=args.sample_rate
    )
    return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    return train_loader, dev_loader


def compute_metrics(args, model, data):
    # Load input, sensitivity maps, and target images onto device
    input, maps, L, R, target, mean, std, norm = data
    input = input.to(args.device)
    maps = maps.to(args.device)
    L = L.to(args.device).squeeze(0)
    R = R.to(args.device).squeeze(0)
    target = target.to(args.device)
    mean = mean.to(args.device)
    std = std.to(args.device)

    # Forward pass through network
    output, _ = model(input, maps, initial_guess=(L, R))

    # Undo normalization from pre-processing
    output = output*std + mean
    target = target*std + mean
    scale = cplx.abs(target).max()

    # Compute image quality metrics from complex-valued images
    cplx_error = cplx.abs(output - target)
    cplx_l1 = torch.mean(cplx_error)
    cplx_l2 = torch.sqrt(torch.mean(cplx_error**2))
    cplx_psnr = 20 * torch.log10(scale / cplx_l2)

    # Compute image quality metrics from magnitude images
    mag_error = torch.abs(cplx.abs(output) - cplx.abs(target))
    mag_l1 = torch.mean(mag_error)
    mag_l2 = torch.sqrt(torch.mean(mag_error**2))
    mag_psnr = 20 * torch.log10(scale / mag_l2)

    return cplx_l1, cplx_l2, cplx_psnr, mag_psnr


def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    #model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):
        model.train()

        # Compute image quality metrics
        l1_loss, l2_loss, cpsnr, mpsnr = compute_metrics(args, model, data)

        # Choose loss function, then run backprop
        loss = l1_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()

        # Write out summary
        writer.add_scalar('Train_Loss', loss.item(), global_step + iter)
        writer.add_scalar('Train_cPSNR', cpsnr.item(), global_step + iter)
        writer.add_scalar('Train_mPSNR', mpsnr.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )

            # Write images into summary
            visualize(args, global_step + iter, model, data_loader, writer)

        start_iter = time.perf_counter()

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    cpsnr_vals = []
    mpsnr_vals = []

    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            # Compute image quality metrics
            l1_loss, l2_loss, cpsnr, mpsnr = compute_metrics(args, model, data)
            losses.append(l1_loss.item())
            cpsnr_vals.append(cpsnr.item())
            mpsnr_vals.append(mpsnr.item())

        writer.add_scalar('Val_Loss', np.mean(losses), epoch)
        writer.add_scalar('Val_cPSNR', np.mean(cpsnr_vals), epoch)
        writer.add_scalar('Val_mPSNR', np.mean(mpsnr_vals), epoch)

    return np.mean(losses), time.perf_counter() - start


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
            input, maps, L, R, target, mean, std, norm = data
            input = input.to(args.device)
            maps = maps.to(args.device)
            L = L.to(args.device).squeeze(0)
            R = R.to(args.device).squeeze(0)
            target = target.to(args.device)

            # Data dimensions (for my own reference)
            #  image size:  [batch_size, nx,   ny, nt, nmaps, 2]
            #  kspace size: [batch_size, nkx, nky, nt, ncoils, 2]
            #  maps size:   [batch_size, nkx,  ny,  1, ncoils, nmaps, 2]

            # Compute DL recon
            output, summary_data = model(input, maps, initial_guess=(L, R))

            # Get initial guess
            init = summary_data['init_image']

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

            # Save scalars to summary
            for i in range(args.num_grad_steps):
                step_size_L = summary_data['step_size_L_%d' % i]
                writer.add_scalar('step_sizes/L%d' % i, step_size_L.item(), epoch)
                step_size_R = summary_data['step_size_R_%d' % i]
                writer.add_scalar('step_sizes/R%d' % i, step_size_R.item(), epoch)

            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=os.path.join(exp_dir, 'model.pt')
    )
    if is_new_best:
        shutil.copyfile(os.path.join(exp_dir, 'model.pt'), 
                        os.path.join(exp_dir, 'best_model.pt'))


def build_model(args):
    model = UnrolledModel(args).to(args.device)
    return model


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
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
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
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0

    logging.info(args)
    logging.info(model)

    train_loader, dev_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        dev_loss, dev_time = evaluate(args, epoch, model, dev_loader, writer)

        scheduler.step(epoch)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Training script for unrolled MRI recon.")
    # Network parameters
    parser.add_argument('--num-grad-steps', type=int, default=5, help='Number of unrolled iterations')
    parser.add_argument('--num-resblocks', type=int, default=2, help='Number of ResNet blocks per iteration')
    parser.add_argument('--num-features', type=int, default=64, help='Number of ResNet channels')
    parser.add_argument('--kernel-size', type=int, default=3, help='Convolution kernel size')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--fix-step-size', type=bool, default=True, help='Fix unrolled step size')
    parser.add_argument('--circular-pad', type=bool, default=True, help='Flag to turn on circular padding')
    parser.add_argument('--slwin-init', action='store_true', help='If set, will use sliding window initialization.')
    parser.add_argument('--share-weights', action='store_true', help='If set, will use share weights between unrolled iterations.')


    # Data parameters
    parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--sample-rate', type=float, default=1.0, help='Fraction of total volumes to include')
    parser.add_argument('--patch-size', default=64, type=int, help='Resolution of images')
    parser.add_argument('--num-emaps', type=int, default=1, help='Number of ESPIRiT maps')

    # Locally low-rank (LLR) Parameters
    parser.add_argument('--num-basis', type=int, default=8, help='Number of basis functions')
    parser.add_argument('--block-size', type=int, default=16, help='Size of blocks/patches')
    parser.add_argument('--overlapping', action='store_true', help='If set, will use overlapping blocks/patches')

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

    # Miscellaneous parameters
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
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

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
