from torch import cuda
from datasets.svhn import SVHN

location = './data'

def main():
    args = {
        "device": "cuda" if cuda.is_available() else 'cpu',
        "batch_size": 32
    }
    dataset = SVHN(None, location=location, num_workers=6)

if __name__ == '__main__':
    main()