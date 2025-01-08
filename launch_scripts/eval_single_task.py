import os
import json
import torch
from utils import torch_load, get_dataloader, compute_accuracy
from args import parse_arguments
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head

# Configurazione specifica per ogni dataset
data_config = {
    "DTD": {},
    "EuroSAT": {},
    "GTSRB": {},
    "MNIST": {},
    "RESISC45": {},
    "SVHN": {},
}

def evaluate_model(encoder_path, dataset_name, args):
    print(f"Inizio della valutazione per il dataset {dataset_name}")

    # Caricamento dell'encoder e della classification head
    encoder = torch_load(encoder_path, device=args.device)
    head = get_classification_head(args, f"{dataset_name}Val")
    model = ImageClassifier(encoder, head)
    model.to(args.device)
    model.eval()

    # Preparazione dei dataset di training, test e validation
    train_dataset = get_dataset(
        f"{dataset_name}Val",
        preprocess=model.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=4
    )
    test_dataset = get_dataset(
        f"{dataset_name}",
        preprocess=model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=4
    )
    val_dataset = get_dataset(
        f"{dataset_name}Val",
        preprocess=model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=4
    )

    train_loader = get_dataloader(train_dataset, is_train=True, args=args)
    test_loader = get_dataloader(test_dataset, is_train=False, args=args)
    val_loader = get_dataloader(val_dataset, is_train=False, args=args)

    # Calcolo dell'accuratezza
    train_accuracy = compute_accuracy(train_loader, model, args.device)
    test_accuracy = compute_accuracy(test_loader, model, args.device)
    val_accuracy = compute_accuracy(val_loader, model, args.device)

    print(
        f"Dataset: {dataset_name}, "
        f"Training Accuracy: {train_accuracy:.4f}, "
        f"Validation Accuracy: {val_accuracy:.4f}, "
        f"Test Accuracy: {test_accuracy:.4f}"
    )
    return train_accuracy, val_accuracy, test_accuracy

def main():
    args = parse_arguments()
    results = {}

    for dataset_name in data_config.keys():
        encoder_path = os.path.join(args.save, f"{dataset_name}_encoder.pth")
        if not os.path.exists(encoder_path):
            print(f"Encoder non trovato per il dataset {dataset_name}: {encoder_path}")
            continue

        train_accuracy, val_accuracy, test_accuracy = evaluate_model(encoder_path, dataset_name, args)
        results[dataset_name] = {
            "training_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "test_accuracy": test_accuracy
        }

    # Salva i risultati in un file JSON
    results_path = os.path.join(args.save, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Risultati della valutazione salvati in {results_path}")

if __name__ == "__main__":
    main()

