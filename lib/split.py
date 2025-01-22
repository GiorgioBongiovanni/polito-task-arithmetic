from typing import Literal

# type alias and iterable
Split = Literal['train', 'validation', 'test']
SPLITS = ['train', 'validation', 'test']

def split_name(dataset_name: str, split: Split) -> str:
    """Correctly adds Val when needed"""
    if split == 'test':
        return dataset_name
    return f'{dataset_name}Val'