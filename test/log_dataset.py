from modeling import ImageClassifier, ImageEncoder
from args import parse_arguments
from datasets.registry import get_dataset
from datasets.common import get_dataloader
from heads import get_classification_head
from lib.split import split_name
from pprint import pprint

results_location = '/home/umberto/Projects/polito-task-arithmetic/results'

def main():
    args = parse_arguments()
    print(args)
    dataset_name = 'EuroSAT'
    encoder = ImageEncoder(args.model, args.cache_dir, args.openclip_cachedir)
    head = get_classification_head(results_location, args, f"{dataset_name}Val")
    model = ImageClassifier(encoder, head)
    model.freeze_head()


    args.save = results_location
    location, batch_size, num_workers = './data', 32, 4

    dataset = get_dataset(split_name(dataset_name, 'train'), model.train_preprocess, location, batch_size, num_workers)
    balanced = True
    loader = get_dataloader(dataset, True, args, balanced=balanced)

    for batch_index, batch in enumerate(loader):
        data_batch, label_batch = batch
        for offset, labeled_entry in enumerate(zip(data_batch, label_batch)):
            data, label = labeled_entry
            index = batch_size * batch_index + offset
            print(f'Entry #{index} class {label}')
            print(data)
            print()

if __name__ == '__main__':
    main()