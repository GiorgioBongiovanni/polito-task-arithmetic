import torch
import torch.nn as nn
import torch.optim as optim
from utils import torch_save, get_dataloader
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

# Funzione principale per il fine-tuning
def finetune_model(dataset_name, args):
    print(f"Inizio del fine-tuning per il dataset {dataset_name}")

    # Preparazione del modello
    encoder = ImageEncoder(args)  # Caricamento del backbone pre-addestrato CLIP ViT-B/32
    head = get_classification_head(args, f"{dataset_name}Val")  # Caricamento della classification head
    model = ImageClassifier(encoder, head)
    model.freeze_head()  # Congela la classification head

    # Preparazione del dataset
    train_dataset = get_dataset(
        f"{dataset_name}Val",
        preprocess=model.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=4
    )

    train_loader = get_dataloader(train_dataset, is_train=True, args=args)

    # Configurazione dell'ottimizzatore e della funzione di perdita
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    # Training loop
    device = torch.device(args.device)
    print(device)
    model.to(device)

    for epoch in range(data_config[dataset_name]["epochs"]):
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            # Gestisci il formato del batch come una tupla
            inputs, targets = batch[0].to(device), batch[1].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Dataset: {dataset_name}, Epoch [{epoch + 1}/{data_config[dataset_name]['epochs']}], "
                    f"Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Salvataggio del modello alla fine del training
    torch_save(model.image_encoder, f"{args.save}/{dataset_name}_encoder.pth")
    print(f"Fine del fine-tuning per il dataset {dataset_name}\n")


def main():

    args = parse_arguments()

    for dataset_name in data_config.keys():
        finetune_model(dataset_name, args)


if __name__ == "__main__":
    main()

