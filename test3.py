import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from conf import settings
from utils import get_network, get_test_dataloader
from editdistance import eval as levenshtein_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def read_cangjie_mapping(file_path):
    cangjie_mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            cangjie_code = parts[-1]
            cangjie_mapping[label] = cangjie_code
    return cangjie_mapping

def map_labels_to_cangjie(labels, cangjie_mapping):
    return [cangjie_mapping[label.item()] for label in labels]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-mapping', type=str, required=True, help='path to the Cangjie mapping file')
    args = parser.parse_args()

    net = get_network(args)

    test_loader = get_test_dataloader(
        root_dir='data/chinese_char/952_test',
        batch_size=args.b,
        num_workers=4,
        shuffle=False
    )

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    cangjie_mapping = read_cangjie_mapping(args.mapping)

    total_levenshtein_distance = 0.0
    total_length = 0.0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(tqdm(test_loader, desc="Testing")):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                image = image.to(device)
                label = label.to(device)
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')

            output = net(image)
            _, pred = output.topk(1, 1, largest=True, sorted=True)

            pred_labels = map_labels_to_cangjie(pred.squeeze(), cangjie_mapping)
            true_labels = map_labels_to_cangjie(label, cangjie_mapping)

            for pred_label, true_label in zip(pred_labels, true_labels):
                total_levenshtein_distance += levenshtein_distance(pred_label, true_label)
                total_length += len(true_label)

    avg_levenshtein_distance = total_levenshtein_distance / total_length
    accuracy = 1 - avg_levenshtein_distance

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Levenshtein Distance: ", avg_levenshtein_distance)
    print("Accuracy: ", accuracy)
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
