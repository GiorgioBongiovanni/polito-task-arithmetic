from collections import Counter
from torch.utils.data import Dataset, Subset, DataLoader
from pprint import pprint

class DatasetBalanced(Exception):
    pass

def get_balanced_subset(dataset: Dataset) -> Subset:
    """Say you have a dataset with the following number of entries per class 7, 9, 10. 
    This function returns a balanced subset with 7, 7, 7 entries respectively per class"""

    if hasattr(dataset, 'targets'):
        class_labels: list[int] = dataset.targets
    else: 
        class_labels: list[int] = [label for _, label in dataset]
    class_counts = Counter(class_labels)
    if len(set(class_counts.values())) == 1:
        raise DatasetBalanced("Calling get_balanced_subset() on a balanced dataset")
    desired_count: int = min(class_counts.values()) # we want this number of entries for each class
    current_count = {class_label: 0 for class_label in class_counts.keys()}
    selected_indices: list[int] = []
    for index, label in enumerate(class_labels):
        if current_count[label] >= desired_count:
            continue
        selected_indices.append(index)
        current_count[label] += 1
    return Subset(dataset, selected_indices)

def balanceable(batch_size=32, num_workers=16):
    def decorator(cls):
        """For a class which has attributes .train_dataset and .test_dataset 
        adds properties .balanced_train_dataset and .balanced_test_dataset.
        It adds the corresponding loader properties as well."""

        @property
        def balanced_train_dataset(self):
            assert hasattr(self, 'train_dataset'), f'{self} must have train_dataset attribute'
            if hasattr(self, '_balanced_train_dataset'): 
                return self._balanced_train_dataset
            balanced: Dataset
            try:
                balanced = get_balanced_subset(self.train_dataset)    
            except DatasetBalanced:
                balanced = self.train_dataset
            self._balanced_train_dataset = balanced
            return balanced
        
        @property
        def balanced_train_loader(self):
            if not hasattr(self, '_balanced_train_loader'):
                self._balanced_train_loader = DataLoader(
                    self.balanced_train_dataset, 
                    batch_size, 
                    num_workers=num_workers)
            return self._balanced_train_loader
        
        @property
        def balanced_test_dataset(self):
            assert hasattr(self, 'test_dataset'), f'{self} must have test_dataset attribute'
            if hasattr(self, '_balanced_test_dataset'): 
                return self._balanced_test_dataset
            balanced: Dataset
            try:
                balanced = get_balanced_subset(self.test_dataset)    
            except DatasetBalanced:
                balanced = self.test_dataset
            self._balanced_test_dataset = balanced
            return balanced
        
        @property
        def balanced_test_loader(self):
            if not hasattr(self, '_balanced_test_loader'):
                self._balanced_test_loader = DataLoader(
                    self.balanced_test_dataset, 
                    batch_size, 
                    num_workers=num_workers)
            return self._balanced_test_loader
        
        cls.balanced_train_dataset = balanced_train_dataset
        cls.balanced_test_dataset = balanced_test_dataset
        cls.balanced_train_loader = balanced_train_loader
        cls.balanced_test_loader = balanced_test_loader
        return cls
    return decorator