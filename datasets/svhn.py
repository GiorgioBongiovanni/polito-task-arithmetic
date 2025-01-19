import os
import torch
from torchvision.datasets import SVHN as PyTorchSVHN
from torch.utils.data import DataLoader
from lib.balance import get_balanced_subset

class SVHN:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16):

        # to fit with repo conventions for location
        modified_location = os.path.join(location, 'svhn')

        self.train_dataset = PyTorchSVHN(
            root=modified_location,
            download=True,
            split='train',
            transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = PyTorchSVHN(
            root=modified_location,
            download=True,
            split='test',
            transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        # Balance handling
        self.balanced_train_dataset = get_balanced_subset(self.train_dataset)
        self.balanced_train_loader = DataLoader(self.balanced_train_dataset, batch_size=batch_size, num_workers=num_workers)
        self.balanced_test_dataset = get_balanced_subset(self.test_dataset)
        self.balanced_test_loader = DataLoader(self.balanced_test_dataset, batch_size=batch_size, num_workers=num_workers)