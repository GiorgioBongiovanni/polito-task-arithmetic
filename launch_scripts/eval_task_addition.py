import os
import torch
import json
from task_vectors import NonLinearTaskVector
from modeling import ImageClassifier
from args import parse_arguments
from datasets.common import get_dataloader
from utils import compute_accuracy, train_diag_fim_logtr
from datasets.registry import get_dataset
from heads import get_classification_head
from lib.config import DATASET_CONFIG

# Names of relevant datasets
datasets = [x for x in DATASET_CONFIG.keys()]

def evaluate_alpha(base_encoder_path, task_vectors, datasets, args, alpha_values, single_task_metrics):
    best_alpha = 0.0
    best_avg_normalized_accuracy = 0.0
    fisher_log_traces = []

    merged_task_vector = sum(task_vectors)  # Sum of task vectors

    for alpha in alpha_values:
        merged_encoder = merged_task_vector.apply_to(base_encoder_path, scaling_coef=alpha)

        avg_normalized_test_accuracy = 0.0

        for dataset_name, task_vector in zip(datasets, task_vectors):
            # Load multi-task model
            head = get_classification_head(args.save, args, f"{dataset_name}Val")
            model = ImageClassifier(merged_encoder, head)
            model.eval()
            model.to(args.device)

            # Prepare validation set
            val_dataset = get_dataset(
                f"{dataset_name}Val",
                preprocess=model.val_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
                num_workers=4
            )
            val_loader = get_dataloader(val_dataset, is_train=False, args=args)

            # Multi-task accuracy on the validation set
            multi_task_accuracy = compute_accuracy(val_loader, model, args.device)

            # Single-task accuracy from JSON file
            single_task_val_accuracy = single_task_metrics[dataset_name]["validation_accuracy"]

            # Calculate normalized accuracy
            normalized_val_accuracy = multi_task_accuracy / single_task_val_accuracy

            avg_normalized_test_accuracy += normalized_val_accuracy

        # Compute average normalized test accuracy
        avg_normalized_test_accuracy /= len(datasets)

        print(f"Alpha: {alpha}, Avg Normalized Validation Accuracy: {avg_normalized_test_accuracy:.4f}")

        # Update best alpha
        if avg_normalized_test_accuracy > best_avg_normalized_accuracy:
            best_avg_normalized_accuracy = avg_normalized_test_accuracy
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha}, Avg Normalized Validation Accuracy: {best_avg_normalized_accuracy:.4f}")
    return best_alpha


def evaluate_model(merged_encoder, dataset_name, args, single_task_metrics):
    print(f"Evaluating dataset {dataset_name}...")

    # Load classification head
    head = get_classification_head(args.save, args, f"{dataset_name}Val")
    model = ImageClassifier(merged_encoder, head)
    model.eval()
    model.to(args.device)

    # Prepare loaders for training and test sets
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

    if args.balanced:
        train_loader = get_dataloader(train_dataset, is_train=True, balanced=True, args=args)
    else:
        train_loader = get_dataloader(train_dataset, is_train=True, args=args)
        
    test_loader = get_dataloader(test_dataset, is_train=False, args=args)

    # Compute absolute accuracies
    train_accuracy = compute_accuracy(train_loader, model, args.device)
    test_accuracy = compute_accuracy(test_loader, model, args.device)

    # Single-task accuracies from JSON file
    train_single_task_accuracy = single_task_metrics[dataset_name]["training_accuracy"]
    test_single_task_accuracy = single_task_metrics[dataset_name]["test_accuracy"]

    # Compute normalized accuracies
    normalized_train_accuracy = train_accuracy / train_single_task_accuracy
    normalized_test_accuracy = test_accuracy / test_single_task_accuracy

    # Compute Fisher metric
    samples_nr = 2000
    fisher_log_trace = train_diag_fim_logtr(args, model, dataset_name, samples_nr)

    print(
        f"Dataset: {dataset_name}, "
        f"Training Accuracy: {train_accuracy:.4f}, "
        f"Test Accuracy: {test_accuracy:.4f}, "
        f"Normalized Training Accuracy: {normalized_train_accuracy:.4f}, "
        f"Normalized Test Accuracy: {normalized_test_accuracy:.4f}, "
        f"Fisher Log-trace: {fisher_log_trace:.4f}"
    )

    return {
        "training_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "normalized_training_accuracy": normalized_train_accuracy,
        "normalized_test_accuracy": normalized_test_accuracy,
        "fisher_log_trace": fisher_log_trace
    }


def main():
    args = parse_arguments()
    print(args)
    base_encoder_path = os.path.join(args.save, "zeroshot_encoder.pth")

    # Ensure the pre-trained model exists
    if not os.path.exists(base_encoder_path):
        raise FileNotFoundError(f"Pre-trained encoder not found: {base_encoder_path}")

    # Load single-task metrics from JSON file
    single_task_metrics_path = os.path.join(args.save, f"evaluation_results_{args.experiment_name}.json")
    with open(single_task_metrics_path, "r") as f:
        single_task_metrics = json.load(f)

    # Build task vectors for each dataset
    task_vectors = []
    for dataset_name in datasets:
        finetuned_path = os.path.join(args.save, f"{dataset_name}_encoder.pth")
        if not os.path.exists(finetuned_path):
            raise FileNotFoundError(f"Fine-tuned encoder not found for {dataset_name}: {finetuned_path}")
        task_vector = NonLinearTaskVector(base_encoder_path, finetuned_path)
        task_vectors.append(task_vector)

    # Search for the best alpha using the validation set
    alpha_values = [round(x * 0.05, 2) for x in range(21)]
    best_alpha = evaluate_alpha(base_encoder_path, task_vectors, datasets, args, alpha_values, single_task_metrics)

    # Compute final accuracies using the best alpha
    merged_task_vector = sum(task_vectors)
    merged_encoder = merged_task_vector.apply_to(base_encoder_path, scaling_coef=best_alpha)
    results = {"best_alpha": best_alpha}

    total_fisher_log_trace = 0.0
    avg_train_accuracy = 0.0
    avg_test_accuracy = 0.0
    avg_normalized_train_accuracy = 0.0
    avg_normalized_test_accuracy = 0.0

    for dataset_name in datasets:
        dataset_results = evaluate_model(merged_encoder, dataset_name, args, single_task_metrics)
        results[dataset_name] = dataset_results

        avg_train_accuracy += dataset_results["training_accuracy"]
        avg_test_accuracy += dataset_results["test_accuracy"]
        avg_normalized_train_accuracy += dataset_results["normalized_training_accuracy"]
        avg_normalized_test_accuracy += dataset_results["normalized_test_accuracy"]
        total_fisher_log_trace += dataset_results["fisher_log_trace"]

    # Compute global averages
    avg_train_accuracy /= len(datasets)
    avg_test_accuracy /= len(datasets)
    avg_normalized_train_accuracy /= len(datasets)
    avg_normalized_test_accuracy /= len(datasets)
    avg_fisher_log_trace = total_fisher_log_trace / len(datasets)

    # Add global averages to results
    results["average"] = {
        "absolute_training_accuracy": avg_train_accuracy,
        "absolute_test_accuracy": avg_test_accuracy,
        "normalized_training_accuracy": avg_normalized_train_accuracy,
        "normalized_test_accuracy": avg_normalized_test_accuracy,
        "fisher_log_trace": avg_fisher_log_trace
    }

    # Print global averages
    print("\n--- Average Results ---")
    print(f"Best Alpha: {best_alpha}")
    print(f"Avg Absolute Training Accuracy: {avg_train_accuracy:.4f}")
    print(f"Avg Absolute Test Accuracy: {avg_test_accuracy:.4f}")
    print(f"Avg Normalized Training Accuracy: {avg_normalized_train_accuracy:.4f}")
    print(f"Avg Normalized Test Accuracy: {avg_normalized_test_accuracy:.4f}")
    print(f"Avg Fisher Log-trace: {avg_fisher_log_trace:.4f}")

    # Save results to JSON file
    results_path = os.path.join(args.save, f"task_addition_results_{args.experiment_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()

