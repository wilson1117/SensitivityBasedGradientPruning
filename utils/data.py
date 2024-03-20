import sys
sys.path.append('..')

import datasets
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from functools import partial
from utils import config
import json

def load_subsets(config_file):
    data_config = json.load(open(config_file))
    dataset_config = config.get_config(data_config['name'])

    dataset = datasets.load_dataset(data_config['name'])[data_config['category']]

    if dataset_config['feature_type'] == 'img':
        transform = dataset_config.get("feature_transforms")
        if transform is not None:
            dataset.set_transform(partial(transform_img, transform=transform, key=dataset_config['img_key']))
    
    public_datasets = Subset(dataset, data_config['public_indices'])
    val_datasets = Subset(dataset, data_config['val_indices'])
    client_datasets = [Subset(dataset, indices) for indices in data_config['client_indices']]

    return (public_datasets, val_datasets, client_datasets), dataset_config

def load_dataset(name, category):
    dataset = datasets.load_dataset(name)[category]
    dataset_config = config.get_config(name)

    if dataset_config['feature_type'] == 'img':
        transform = dataset_config.get("feature_transforms")
        if transform is not None:
            dataset.set_transform(partial(transform_img, transform=transform, key=dataset_config['img_key']))

    return dataset, dataset_config

def to_dataloader(datasets, batch_size, shuffle=True, num_workers=0):
    if type(datasets) is list:
        return [DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers) for dataset in datasets]
    
    return DataLoader(datasets, batch_size, shuffle=shuffle, num_workers=num_workers)


def transform_img(x, transform, key):
    x[key] = [transform(img) for img in x[key]]
    return x

if __name__ == '__main__':
    load_subsets('config\cifar10_train_20_1_42.json')