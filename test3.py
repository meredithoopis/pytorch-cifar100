import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from conf import settings
from utils import get_network, get_test_dataloader
from editdistance import eval as levenshtein_distance
import time
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def read_cangjie_mapping(file_path):
    cangjie_mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip the header row
        for line in f:
            parts = line.strip().split()
            character = parts[1]
            label = int(parts[0])
            cangjie_code = parts[-1]
            cangjie_mapping[label] = (cangjie_code, character)
    return cangjie_mapping

mappings = read_cangjie_mapping('chinese-char/etl_952_singlechar_size_64/952_labels.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args).to(device)

    test_loader = get_test_dataloader(
        root_dir='chinese-char/etl_952_singlechar_size_64/952_test',
        batch_size=args.b,
        num_workers=4,
        shuffle=False
    )

    net.load_state_dict(torch.load(args.weights, map_location=device))
    print(net)
    net.eval()

    total_levenshtein_distance = 0.0
    total_length = 0.0
    test_loss = 0.0
    loss_function = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        start = time.time()
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            if batch_idx == 0:
                print(f"First batch - images shape: {images.shape}, labels shape: {labels.shape}")

            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()

            _, preds = outputs.max(1)

            for pred, label in zip(preds, labels):
                pred_code = mappings.get(pred.item())
                label_code = mappings.get(label.item())

                if pred_code is None or label_code is None:
                    print(f"Mapping missing for pred: {pred.item()} or label: {label.item()}")
                    continue

                levenshtein_distance_value = levenshtein_distance(pred_code[0], label_code[0])
                total_levenshtein_distance += levenshtein_distance_value
                total_length += len(label_code[0]) if label_code[0] != 'zc' else 1

                if batch_idx == 0 and pred.item() == 0:
                    print(f"Sample - pred: {pred.item()}, label: {label.item()}, pred_code: {pred_code}, label_code: {label_code}")

    if total_length == 0:
        print("No data processed. Please check the data loader and ensure there are valid samples in the dataset.")
    else:
        avg_levenshtein_distance = total_levenshtein_distance / total_length
        accuracy = 1 - avg_levenshtein_distance

        finish = time.time()

        if args.gpu:
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')

        print("Testing Network.....")
        print('Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed: {:.2f}s'.format(
            test_loss / len(test_loader.dataset),
            accuracy,
            finish - start
        ))

    print("Levenshtein Distance: ", avg_levenshtein_distance if total_length != 0 else "N/A")
    print("Accuracy: ", accuracy if total_length != 0 else "N/A")


 

 





