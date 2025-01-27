from args import parse_arguments
from modeling import ImageEncoder
from pprint import pprint
from argparse import Namespace
from datasets.registry import get_dataset
from datasets.common import get_dataloader
from torch.utils.data import DataLoader
from collections import Counter
from lib.split import SPLITS, split_name

def print_loader_class_distribution(loader: DataLoader):
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.tolist())
    class_frequencies = Counter(all_labels)
    pprint(class_frequencies)

def verify_balancing(dataset_name: str, args: Namespace, encoder: ImageEncoder):
    data_location, batch_size = args.data_location, args.batch_size
    train_preprocess, val_preprocess = encoder.train_preprocess, encoder.val_preprocess

    for split in SPLITS:
        spl_name = split_name(dataset_name, split)
        preprocess = train_preprocess if split == 'train' else val_preprocess

        for balanced in [False, True]:
            print(f'{dataset_name=}, {split=}, forcing balance: {balanced}')
            dataset = get_dataset(spl_name, preprocess, data_location, num_workers=4)
            loader = get_dataloader(dataset, is_train=split=='train', args=args, balanced=balanced)
            print_loader_class_distribution(loader)

def main():
    args: Namespace = parse_arguments()
    args.__dict__['data_location'] = './data'
    encoder = ImageEncoder(args)

    for dataset_name in ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]:
        verify_balancing(dataset_name, args, encoder)

if __name__ == '__main__':
    main()