"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random
import sys
import numpy as np
import h5py
from torch.utils.data import Dataset


class SliceData(Dataset):
    """
    A generic PyTorch Dataset class that provides access to 2D MR image slices.
    """

    def __init__(self, root, transform, sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """

        self.transform = transform

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
#             kspace = h5py.File(fname, 'r')['Input']
            self.examples += [(fname)]
#         print(self.examples)
#         sys.exit()
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname = self.examples[i]
        with h5py.File(fname, 'r') as data:
            MRF = np.array(data['MRF'])
            T1 = np.array(data['T1'])
            T2 = np.array(data['T2'])
            FLAIR = np.array(data['FLAIR'])
            MRF_avg = np.array(data['MRF_avg'])
#             maps = data['maps'][...,slice]
            return self.transform(MRF, T1,T2,FLAIR,MRF_avg)




