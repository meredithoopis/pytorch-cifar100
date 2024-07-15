import argparse
import torch
from tqdm import tqdm
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
            label = int(parts[0])
            character = parts[1]
            cangjie_code = parts[-1]
            cangjie_mapping[label] = (cangjie_code, character)
    return cangjie_mapping

mappings = read_cangjie_mapping('chinese-char/etl_952_singlechar_size_64/952_labels.txt')

def inference(net, weights_path):
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.eval()

    test_loader = get_test_dataloader(
        root_dir='chinese-char/etl_952_singlechar_size_64/952_test',
        batch_size=16,
        num_workers=2,  
        shuffle=False
    )

    outs = []
    with torch.no_grad():
        start = time.time()
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, preds = outputs.max(1)

            for pred, label in zip(preds, labels):
                if pred.item() in mappings:
                    pred_code = mappings[pred.item()][0]
                    character = mappings[pred.item()][1]
                    if pred_code == 'zc':
                        outs.append(character)
                    else:
                        outs.append(pred_code)
                else:
                    print(f"Mapping missing for pred: {pred.item()}")

    return outs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    args = parser.parse_args()

    net = get_network(args).to(device)

    result = inference(net, args.weights)

    print(result)
