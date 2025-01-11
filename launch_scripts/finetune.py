import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import torch_save, get_dataloader, compute_accuracy, compute_fisher_log_trace
from args import parse_arguments
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head

# Configurazione specifica per ogni dataset
data_config = {
    "DTD": {"epochs": 76},
    "EuroSAT": {"epochs": 12},
    "GTSRB": {"epochs": 11},
    "MNIST": {"epochs": 5},
    "RESISC45": {"epochs": 15},
    "SVHN": {"epochs": 4},
}


def finetune_model(dataset_name, args):
    print(f"Inizio del fine-tuning per il dataset {dataset_name}")

    # Preparazione del modello
    encoder = ImageEncoder(args)
    head = get_classification_head(args, f"{dataset_name}Val")
    model = ImageClassifier(encoder, head)
    model.freeze_head()

    # Preparazione del dataset
    train_dataset = get_dataset(
        f"{dataset_name}Val",
        preprocess=model.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=4,
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

    # Calcolo dell'accuratezza del modello pre-addestrato
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    train_accuracy = compute_accuracy(train_loader, model, device)
    test_accuracy = compute_accuracy(test_loader, model, device)
    print(f"Pre-trained Model Accuracy sul dataset {dataset_name}:")
    print(f" - Training Accuracy: {train_accuracy:.4f}")
    print(f" - Test Accuracy: {test_accuracy:.4f}")

    # Calcolo della metrica di Fisher
    fisher_log_trace = compute_fisher_log_trace(args, model, dataset_name)
    print(f" - Fisher Log-Trace: {fisher_log_trace:.4f}")

    # Configurazione dell'ottimizzatore e della funzione di perdita
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    model.train()
    for epoch in range(data_config[dataset_name]["epochs"]):
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"Dataset: {dataset_name}, Epoch [{epoch + 1}/{data_config[dataset_name]['epochs']}], "
                    f"Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

    # Salvataggio del modello alla fine del training
    torch_save(model.image_encoder, f"{args.save}/{dataset_name}_encoder.pth")
    print(f"Fine del fine-tuning per il dataset {dataset_name}\n")

    return train_accuracy, test_accuracy, fisher_log_trace


def main():
    args = parse_arguments()
    print(f"\nExperiment name = {args.experiment_name}\n")

    # Salva il modello pre-addestrato una volta prima di iniziare il fine-tuning
    encoder = ImageEncoder(args)
    pretrain_path = os.path.join(args.save, "zeroshot_encoder.pth")
    if not os.path.exists(pretrain_path):
        torch_save(encoder, pretrain_path)
        print(f"Salvato il modello pre-addestrato in {pretrain_path}")

    total_train_accuracy = 0.0
    total_test_accuracy = 0.0
    total_fisher_log_trace = 0.0
    dataset_count = len(data_config)

    for dataset_name in data_config.keys():
        train_accuracy, test_accuracy, fisher_log_trace = finetune_model(dataset_name, args)
        total_train_accuracy += train_accuracy
        total_test_accuracy += test_accuracy
        total_fisher_log_trace += fisher_log_trace

    avg_train_accuracy = total_train_accuracy / dataset_count
    avg_test_accuracy = total_test_accuracy / dataset_count
    avg_fisher_log_trace = total_fisher_log_trace / dataset_count

    print("\n--- Average Results ---")
    print(f"Avg Training Accuracy: {avg_train_accuracy:.4f}")
    print(f"Avg Test Accuracy: {avg_test_accuracy:.4f}")
    print(f"Avg Fisher Log-Trace: {avg_fisher_log_trace:.4f}")


if __name__ == "__main__":
    main()



