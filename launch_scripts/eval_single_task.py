import os
import json
import torch
from utils import torch_load, get_dataloader, compute_accuracy, train_diag_fim_logtr
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

    if args.balanced:
        train_loader = get_dataloader(train_dataset, is_train=True, balanced=True, args=args)
    else:
        train_loader = get_dataloader(train_dataset, is_train=True, args=args)
        
    test_loader = get_dataloader(test_dataset, is_train=False, args=args)
    val_loader = get_dataloader(val_dataset, is_train=False, args=args)

    # Calcolo dell'accuratezza
    train_accuracy = compute_accuracy(train_loader, model, args.device)
    test_accuracy = compute_accuracy(test_loader, model, args.device)
    val_accuracy = compute_accuracy(val_loader, model, args.device)

    # Calcolo della metrica sulla matrice di Fisher
    samples_nr = 2000
    log_trace_fisher = train_diag_fim_logtr(args, model, dataset_name, samples_nr)

    print(
        f"Dataset: {dataset_name}, "
        f"Training Accuracy: {train_accuracy:.4f}, "
        f"Validation Accuracy: {val_accuracy:.4f}, "
        f"Test Accuracy: {test_accuracy:.4f}, "
        f"Log-trace Fisher: {log_trace_fisher:.4f}"
    )

    return train_accuracy, val_accuracy, test_accuracy, log_trace_fisher

def main():
    args = parse_arguments()
    results = {}
    total_train_accuracy = 0.0
    total_test_accuracy = 0.0
    total_fisher_log_trace = 0.0
    dataset_count = len(data_config)

    for dataset_name in data_config.keys():
        encoder_path = os.path.join(args.save, f"{dataset_name}_encoder.pth")
        if not os.path.exists(encoder_path):
            print(f"Encoder non trovato per il dataset {dataset_name}: {encoder_path}")
            continue

        train_accuracy, val_accuracy, test_accuracy, log_trace_fisher = evaluate_model(encoder_path, dataset_name, args)
        results[dataset_name] = {
            "training_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "log_trace_fisher": log_trace_fisher
        }
        total_train_accuracy += train_accuracy
        total_test_accuracy += test_accuracy
        total_fisher_log_trace += log_trace_fisher

    avg_train_accuracy = total_train_accuracy / dataset_count
    avg_test_accuracy = total_test_accuracy / dataset_count
    avg_fisher_log_trace = total_fisher_log_trace / dataset_count

    print(f"\n--- Average Results ---")
    print(f"Avg Training Accuracy: {avg_train_accuracy:.4f}")
    print(f"Avg Test Accuracy: {avg_test_accuracy:.4f}")
    print(f"Avg Log-trace Fisher: {avg_fisher_log_trace:.4f}")

    results["average"] = {
        "training_accuracy": avg_train_accuracy,
        "test_accuracy": avg_test_accuracy,
        "log_trace_fisher": avg_fisher_log_trace
    }

    # Salva i risultati in un file JSON
    results_path = os.path.join(args.save, f"evaluation_results_{args.experiment_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Risultati della valutazione salvati in {results_path}")

if __name__ == "__main__":
    main()
