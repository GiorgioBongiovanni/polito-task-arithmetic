import os
import torch
import json
from task_vectors import NonLinearTaskVector
from modeling import ImageClassifier
from args import parse_arguments
from utils import torch_load, get_dataloader
from datasets.registry import get_dataset
from heads import get_classification_head
from typing import Literal
from modeling import ImageClassifier

# dataset specific configuration
datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]


def compute_accuracy(loader: torch.utils.data.DataLoader, model: ImageClassifier, device: Literal['cpu', 'gpu']):
    """Returns correctly labeled entries over the total of tested entries"""
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return correct / total


def evaluate_model(merged_encoder, dataset_name, args, task_vector, base_encoder_path, alpha):
    print(f"Valutazione sul dataset {dataset_name}...")

    # Caricamento della classification head specifica
    head = get_classification_head(args, f"{dataset_name}Val")
    model = ImageClassifier(merged_encoder, head)
    model.eval()
    model.to(args.device)

    # Preparazione dei loader per training e test
    train_dataset = get_dataset(
        f"{dataset_name}Val",
        preprocess=model.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=4
    )
    test_dataset = get_dataset(
        dataset_name,
        preprocess=model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=4
    )
    train_loader = get_dataloader(train_dataset, is_train=True, args=args)
    test_loader = get_dataloader(test_dataset, is_train=False, args=args)

    # Calcolo delle accuratezze assolute
    train_accuracy = compute_accuracy(train_loader, model, args.device)
    test_accuracy = compute_accuracy(test_loader, model, args.device)

    # Calcolo delle accuratezze normalizzate
    single_task_encoder = task_vector.apply_to(base_encoder_path, scaling_coef=alpha)
    single_task_model = ImageClassifier(single_task_encoder, head)
    single_task_model.eval()
    single_task_model.to(args.device)

    train_single_task_accuracy = compute_accuracy(train_loader, single_task_model, args.device)
    test_single_task_accuracy = compute_accuracy(test_loader, single_task_model, args.device)

    normalized_train_accuracy = train_accuracy / train_single_task_accuracy
    normalized_test_accuracy = test_accuracy / test_single_task_accuracy

    return {
        "training_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "normalized_training_accuracy": normalized_train_accuracy,
        "normalized_test_accuracy": normalized_test_accuracy
    }


def evaluate_alpha(base_encoder_path, task_vectors, datasets, args, alpha_values):
    best_alpha = 0.0
    best_avg_normalized_accuracy = 0.0

    # Iterazione su tutti i valori di alpha
    merged_task_vector = sum(task_vectors)  # Somma dei task vector
    for alpha in alpha_values:
        merged_encoder = merged_task_vector.apply_to(base_encoder_path, scaling_coef=alpha)  # Applicazione del task vector scalato

        avg_normalized_accuracy = 0.0

        # Iterazione sui dataset per calcolare l'accuratezza normalizzata sul validation set
        for dataset_name, task_vector in zip(datasets, task_vectors):
            head = get_classification_head(args, f"{dataset_name}Val")
            model = ImageClassifier(merged_encoder, head)
            model.eval()
            model.to(args.device)

            # Caricamento del validation set
            val_dataset = get_dataset(
                f"{dataset_name}Val",
                preprocess=model.val_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
                num_workers=4
            )
            val_loader = get_dataloader(val_dataset, is_train=False, args=args)

            # Accuratezza del modello multi-task sul validation set
            multi_task_accuracy = compute_accuracy(val_loader, model, args.device)

            # Accuratezza del modello single-task sul validation set
            single_task_model = task_vector.apply_to(base_encoder_path, scaling_coef=alpha)
            single_task_model = ImageClassifier(single_task_model, head)
            single_task_model.eval()
            single_task_model.to(args.device)
            single_task_accuracy = compute_accuracy(val_loader, single_task_model, args.device)

            # Accuratezza normalizzata
            normalized_accuracy = multi_task_accuracy / single_task_accuracy
            avg_normalized_accuracy += normalized_accuracy

        # Calcolo dell'accuratezza media normalizzata
        avg_normalized_accuracy /= len(datasets)

        print(f"Alpha: {alpha}, Avg Normalized Validation Accuracy: {avg_normalized_accuracy:.4f}")

        # Aggiornamento del miglior alpha
        if avg_normalized_accuracy > best_avg_normalized_accuracy:
            best_avg_normalized_accuracy = avg_normalized_accuracy
            best_alpha = alpha

    print(f"\nMiglior alpha: {best_alpha}, Avg Normalized Validation Accuracy: {best_avg_normalized_accuracy:.4f}")
    return best_alpha



def main():
    args = parse_arguments()
    base_encoder_path = os.path.join(args.save, "zeroshot_encoder.pth")
    
    # Controllo che il modello pre-addestrato esista
    if not os.path.exists(base_encoder_path):
        raise FileNotFoundError(f"Encoder pre-addestrato non trovato: {base_encoder_path}")

    # Costruzione dei task vector per ogni dataset
    task_vectors = []
    for dataset_name in datasets:
        finetuned_path = os.path.join(args.save, f"{dataset_name}_encoder.pth")
        if not os.path.exists(finetuned_path):
            raise FileNotFoundError(f"Encoder fine-tuned non trovato per {dataset_name}: {finetuned_path}")
        task_vector = NonLinearTaskVector(base_encoder_path, finetuned_path)
        task_vectors.append(task_vector)

    # Ricerca del miglior alpha usando il validation set
    alpha_values = [round(x * 0.05, 2) for x in range(21)]
    best_alpha = evaluate_alpha(base_encoder_path, task_vectors, datasets, args, alpha_values)
    # best_alpha = 0.3

    # Calcolo delle accuratezze finali usando alpha ottimale
    merged_task_vector = sum(task_vectors)
    merged_encoder = merged_task_vector.apply_to(base_encoder_path, scaling_coef=best_alpha)
    results = {}
    avg_absolute_train_accuracy = 0.0
    avg_absolute_test_accuracy = 0.0
    avg_normalized_train_accuracy = 0.0
    avg_normalized_test_accuracy = 0.0

    for dataset_name, task_vector in zip(datasets, task_vectors):
        dataset_results = evaluate_model(merged_encoder, dataset_name, args, task_vector, base_encoder_path, best_alpha)
        results[dataset_name] = dataset_results

        # Stampa accuracies per ogni dataset
        print(f"Dataset: {dataset_name}")
        print(f" - Training Accuracy: {dataset_results['training_accuracy']:.4f}")
        print(f" - Test Accuracy: {dataset_results['test_accuracy']:.4f}")
        print(f" - Normalized Training Accuracy: {dataset_results['normalized_training_accuracy']:.4f}")
        print(f" - Normalized Test Accuracy: {dataset_results['normalized_test_accuracy']:.4f}")

        avg_absolute_train_accuracy += dataset_results['training_accuracy']
        avg_absolute_test_accuracy += dataset_results['test_accuracy']
        avg_normalized_train_accuracy += dataset_results['normalized_training_accuracy']
        avg_normalized_test_accuracy += dataset_results['normalized_test_accuracy']

    avg_absolute_train_accuracy /= len(datasets)
    avg_absolute_test_accuracy /= len(datasets)
    avg_normalized_train_accuracy /= len(datasets)
    avg_normalized_test_accuracy /= len(datasets)

    print("\n--- Average Results ---")
    print(f"Avg Absolute Training Accuracy: {avg_absolute_train_accuracy:.4f}")
    print(f"Avg Absolute Test Accuracy: {avg_absolute_test_accuracy:.4f}")
    print(f"Avg Normalized Training Accuracy: {avg_normalized_train_accuracy:.4f}")
    print(f"Avg Normalized Test Accuracy: {avg_normalized_test_accuracy:.4f}")

    results["average"] = {
        "absolute_training_accuracy": avg_absolute_train_accuracy,
        "absolute_test_accuracy": avg_absolute_test_accuracy,
        "normalized_training_accuracy": avg_normalized_train_accuracy,
        "normalized_test_accuracy": avg_normalized_test_accuracy
    }

    # Salvataggio dei risultati
    results_path = os.path.join(args.save, "task_addition_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Risultati salvati in {results_path}")



if __name__ == "__main__":
    main()

