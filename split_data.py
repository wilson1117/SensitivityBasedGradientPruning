import argparse
import numpy as np
import datasets
from tqdm import tqdm
import os
import json
from utils import config
import random

def split_data(name, category, num_client, classes_per_client, public_set, val_set, seed, args):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    data_config = config.get_config(name)
    dataset = datasets.load_dataset(name)[category]

    num_classes = data_config['num_classes']
    class_indices = [[] for _ in range(num_classes)]

    print('Counting classes...')
    for i, item in enumerate(tqdm(dataset)):
        class_indices[data_config['feature_extractor'](item)[1]].append(i)

    public_per_class = public_set // num_classes
    val_per_class = val_set // num_classes

    public_indices = []
    val_indices = []
    for idx in range(num_classes):
        np.random.shuffle(class_indices[idx])
        public_indices.extend(class_indices[idx][:public_per_class])
        val_indices.extend(class_indices[idx][public_per_class : public_per_class + val_per_class])
        class_indices[idx] = class_indices[idx][public_per_class + val_per_class:]

    class_indices = np.concatenate(class_indices)

    num_shard = num_client * classes_per_client
    shards = np.array_split(class_indices, num_shard)
    shards = [(data_config['feature_extractor'](dataset[shard[0].item()])[1], shard) for shard in shards]
    
    while True:
        client_indices = []
        bufferd_shards = [*shards]

        for _ in range(num_client):
            selected_classes = []
            selected_shards = []
            random.shuffle(bufferd_shards)

            for i in range(len(bufferd_shards)):
                idx = i - len(selected_shards)
                if bufferd_shards[idx][0] not in selected_classes:
                    selected_classes.append(bufferd_shards[idx][0])
                    selected_shards.append(bufferd_shards[idx][1])
                    bufferd_shards.pop(idx)
                    if len(selected_classes) == classes_per_client:
                        break
            else:
                break

            client_indices.append(np.concatenate(selected_shards).tolist())
        else:
            break

    return {
        'name': name,
        'category': category,
        'num_client': num_client,
        'params': vars(args),
        'public_indices': public_indices,
        'val_indices': val_indices,
        'client_indices': client_indices
    }

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--dataset', type=str, default='cifar10')
    argparser.add_argument('--category', type=str, default='train', help='category of dataset')
    argparser.add_argument('--num-client', type=int, default=100, help='Number of clients')
    argparser.add_argument('--classes-per-client', type=int, default=2, help='Number of classes per client')
    argparser.add_argument('--public-set', type=int, default=100, help='Number of public set')
    argparser.add_argument('--val-set', type=int, default=9900, help='Number of validation set')
    argparser.add_argument('--seed', type=int, default=42, help='Random seed')
    argparser.add_argument('--out-dir', type=str, default='data_config', help='Output directory')

    args = argparser.parse_args()

    config = split_data(args.dataset, args.category, args.num_client, args.classes_per_client, args.public_set, args.val_set, args.seed, args)
    filename = f'{args.dataset}_{args.num_client}_{args.classes_per_client}_{args.seed}.json'
    filepath = os.path.join(args.out_dir, filename)

    os.makedirs(args.out_dir, exist_ok=True)

    with open(filepath, 'w') as file:
        json.dump(config, file)

