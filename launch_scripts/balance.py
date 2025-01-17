from utils import get_dataset, get_dataloader
from torch import cuda
from torch.utils.data import Dataset, Subset
from datasets.dtd import DTD
from datasets.eurosat import EuroSAT
from datasets.gtsrb import GTSRB
from collections import Counter
import numpy as np

location = './data'

class DatasetBalanced(Exception):
    pass

def get_balanced_subset(dataset: Dataset) -> Subset:
    """Say you have a dataset with the following number of entries per class 7, 9, 10. 
    This function returns a balanced subset with 7, 7, 7 entries respectively per class"""

    if hasattr(dataset, 'targets'):
        class_labels: list[int] = dataset.targets
    else: 
        class_labels: list[int] = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(class_labels)
    if len(set(class_counts.values())) == 1:
        raise DatasetBalanced("Calling get_balanced_subset() on a balanced dataset")
    min_count: int = min(class_counts.values())
    np_class_labels = np.asarray(class_labels, dtype=np.int16)
    subsamplig_mask = np.zeros_like(np_class_labels, dtype=bool)

    for class_label in class_counts.keys():
        class_indices = np.where(np_class_labels == class_label)[0]
        selected_indices = np.random.choice(
            class_indices, 
            size=min_count, 
            replace=False
        )
        subsamplig_mask[selected_indices] = True
    return Subset(dataset, subsamplig_mask)

def main():
    args = {
        "device": "cuda" if cuda.is_available() else 'cpu',
        "batch_size": 32
    }
    dataset = GTSRB(None, location=location)
    train_dataset = dataset.train_dataset
    subset = get_balanced_subset(train_dataset)

if __name__ == '__main__':
    main()