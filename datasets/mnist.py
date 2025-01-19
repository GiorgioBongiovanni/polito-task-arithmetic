import os
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from lib.balance import get_balanced_subset

class MNIST:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16):


        self.train_dataset = datasets.MNIST(
            root=location,
            download=True,
            train=True,
            transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = datasets.MNIST(
            root=location,
            download=True,
            train=False,
            transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        # Balance handling. The dataset is already balanced, no need to subsample
        self.balanced_train_dataset = self.train_dataset
        self.balanced_train_loader = self.train_loader
        self.balanced_test_dataset = self.test_dataset
        self.balanced_test_loader = self.test_loader