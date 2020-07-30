"""
Deep subspace learning network.
by Christopher M. Sandino (sandino@stanford.edu), 2020.

"""

import torch
from torch import nn
import utils.complex_utils as cplx
from utils.transforms import SenseModel
from utils.transforms import ArrayToBlocks
import utils.layers2D.ResNet as ResNet


class RNN(nn.Module):
    """
    Prototype for long short-term memory (LSTM) network.
    """

    def __init__(self, in_chans, hidden_size, num_layers, bidirectional=True):
        """

        """
        super().__init__()

        num_directions = 2 if bidirectional is True else 1

        self.rnn = nn.LSTM(
            input_size=in_chans,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # So that that the output has the same number of channels as the input
        # TODO: make this multi-layer?
        self.resample = nn.Linear(hidden_size*num_directions, in_chans)


    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, time, in_chans]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, time, in_chans]
        """

        output, _ = self.rnn(input, None) # None represents zero initial hidden state

        return self.resample(output)


class UnrolledModel(nn.Module):
    """
    PyTorch implementation of Deep Subspace Learning Reconstruction (DSLR) Network.
    """

    def __init__(self, params):
        """
        Args:
            params (dict): Dictionary containing network parameters
        """
        super().__init__()

        # Network parameters
        self.num_altmin_steps = params.num_grad_steps 
        num_resblocks = params.num_resblocks
        num_features = params.num_features
        drop_prob = params.drop_prob
        #circular_pad = params.circular_pad
        fix_step_size = params.fix_step_size
        kernel_size = params.kernel_size
        self.num_emaps = params.num_emaps

        # LLR parameters
        self.num_basis = params.num_basis
        self.block_size = params.block_size
        self.overlapping = params.overlapping

        # Spatial ResNet parameters
        sp_net_params = dict(num_resblocks=num_resblocks,
                             in_chans=2*self.num_basis*self.num_emaps,
                             chans=num_features,
                             kernel_size=(kernel_size, kernel_size),
                             drop_prob=drop_prob,
                             circular_pad=False
                            )

        # Temporal ResNet paramters
        t_net_params = dict(num_resblocks=num_resblocks,
                            in_chans=2*self.num_basis,
                            chans=num_features,
                            kernel_size=(1, kernel_size),
                            drop_prob=drop_prob,
                            circular_pad=True
                            )

        # Temporal RNN paramters
        rnn_params = dict(in_chans=2*self.num_basis,
                          hidden_size=num_features,
                          num_layers=3,
                          bidirectional=True
                         )

        # Declare ResNets and RNNs for each alt min iteration
        if share_weights:
            self.sp_resnets = nn.ModuleList([ResNet(**sp_net_params)] * self.num_altmin_steps)
            self.t_resnets = nn.ModuleList([ResNet(**t_net_params)] * self.num_altmin_steps)
            #self.rnns = nn.ModuleList([RNN(**rnn_params)] * self.num_altmin_steps)
        else:
            self.sp_resnets = nn.ModuleList([ResNet(**sp_net_params) for i in range(self.num_altmin_steps)])
            self.t_resnets = nn.ModuleList([ResNet(**t_net_params) for i in range(self.num_altmin_steps)])
            #self.rnns = nn.ModuleList([RNN(**rnn_params) for i in range(self.num_altmin_steps)])

        # Declare ResNets for each unrolled iteration
        self.sp_resnets = nn.ModuleList([ResNet(**sp_net_params) for i in range(self.num_altmin_steps)])
        self.t_resnets = nn.ModuleList([ResNet(**t_net_params) for i in range(self.num_altmin_steps)])


    def reshape_LR(self, L, L_shape, R, R_shape, beforeNet=False):
        # Get L, R dimensions
        np, nxp, nyp,  _, ne, nb, _ = L_shape
        np,   _,   _, nt,  _, nb, _ = R_shape
        L_net_shape = (np, ne*nb*2, nyp, nxp)
        R_net_shape = (np, nb*2,      1,  nt)

        if beforeNet:
            L = L.permute(0,4,5,6,3,2,1).reshape(L_net_shape)
            R = R.permute(0,4,5,6,3,2,1).reshape(R_net_shape)
        else:
            L = L.permute(0,3,2,1).reshape(L_shape)
            R = R.permute(0,3,2,1).reshape(R_shape)

        return L, R


    def compose_LR(self, L, R, block_op):
        patches = torch.sum(cplx.mul(L, cplx.conj(R)), dim=-2)
        return block_op(patches, adjoint=True)


    def get_step_sizes(self, L, R, alpha=0.9):
        """
        Computes step size based on eigenvalues of L'*L and R'*R.
        """
        # Get data dimensions
        np, nxp, nyp,  _, ne, nb, _ = L.shape
        np,   _,   _, nt,  _, nb, _ = R.shape

        # Compute L step size (based on R)
        R = R.reshape((np, nt, nb, 2))
        E = cplx.power_method(R, num_iter=10)
        step_size_L = -1.0 * alpha / E.max()

        # Compute R step size (based on L)
        L = L.reshape((np, nxp*nyp*ne, nb, 2))
        E = cplx.power_method(L, num_iter=10)
        step_size_R = -1.0 * alpha / E.max()

        return step_size_L, step_size_R


    def forward(self, kspace, maps, initial_guess=None, mask=None):
        """
        Args:
            kspace (torch.Tensor): Input tensor of shape [batch_size, height, width, time, num_coils, 2]
            maps (torch.Tensor): Input tensor of shape   [batch_size, height, width,    1, num_coils, num_emaps, 2]
            mask (torch.Tensor): Input tensor of shape   [batch_size, height, width, time, 1, 1]

        Intermediate variables:
            Spatial basis vectors:  [batch_size, block_size, block_size,    1, num_emaps, num_basis, 2]
            Temporal basis vectors: [batch_size,          1,          1, time,         1, num_basis, 2]

        Returns:
            (torch.Tensor): Output tensor of shape       [batch_size, height, width, time, num_emaps, 2]
        """
        summary_data = {}

        if self.num_emaps != maps.size()[-2]:
            raise ValueError('Incorrect number of ESPIRiT maps! Re-prep data...')
        image_shape = kspace.shape[0:4] + (self.num_emaps, 2)

        if mask is None:
            mask = cplx.get_mask(kspace)

        # Declare linear operators
        A = SenseModel(maps, weights=mask)
        BlockOp = ArrayToBlocks(self.block_size, image_shape, overlapping=self.overlapping)

        # Compute zero-filled image reconstruction
        zf_image = A(kspace, adjoint=True)

        # Get initial guess for L, R basis vectors
        if initial_guess is None:
            L, R = decompose_LR(zf_image, block_op=BlockOp)
        else:
            L, R = initial_guess
        image = self.compose_LR(L, R, BlockOp)

        # save into summary
        summary_data['init_image'] = image
        
        # Begin unrolled alternating minimization
        for i, (sp_resnet, t_resnet) in enumerate(zip(self.sp_resnets, self.t_resnets)):
            # Save previous L,R variables
            L_prev = L
            R_prev = R

            # Compute gradients of ||Y - ALR'||_2 w.r.t. L, R
            grad_x = BlockOp(A(A(image), adjoint=True) - zf_image).unsqueeze(-2)
            L = torch.sum(cplx.mul(grad_x, R_prev), keepdim=True, dim=3)
            R = torch.sum(cplx.mul(cplx.conj(grad_x), L_prev), keepdim=True, dim=(1,2,4))

            # L, R model updates
            step_size_L, step_size_R = self.get_step_sizes(L_prev, R_prev)
            L = L_prev + step_size_L * L
            R = R_prev + step_size_R * R

            # L, R network updates
            L, R = self.reshape_LR(L, L_prev.shape, R, R_prev.shape, beforeNet=True)
            L, R = sp_resnet(L), t_resnet(R)
            L, R = self.reshape_LR(L, L_prev.shape, R, R_prev.shape, beforeNet=False)

            # Get current image estimate
            image = self.compose_LR(L, R, BlockOp)

            # Save summary variables
            summary_data['image_%d' % i] = image
            summary_data['step_size_L_%d' % i] = step_size_L
            summary_data['step_size_R_%d' % i] = step_size_R

        return image, summary_data
