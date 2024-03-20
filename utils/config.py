import sys
sys.path.append('..')

from torchvision import transforms
from utils import data

def get_config(dataset_name):
    if dataset_name == 'mnist':
        return {
            'name': 'mnist',
            'feature_type': 'img',
            'label_type': 'onehot',
            'num_classes': 10,
            'feature_transforms': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]),
            'normalize': ((0.1307,), (0.3081,)),
            'feature_extractor': (lambda item: ([item['image'],], item['label'])),
            'metric': 'OneHotMetric',
            'criterion': 'CrossEntropyLoss',
            'grayscale': True,
            'img_key': 'image'
        }
    elif dataset_name == 'cifar10':
        return {
            'name': 'cifar10',
            'feature_type': 'img',
            'label_type': 'onehot',
            'num_classes': 10,
            'feature_transforms': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))
            ]),
            'normalize': ((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)),
            'feature_extractor': (lambda item: ([item['img'],], item['label'])),
            'metric': 'OneHotMetric',
            'criterion': 'CrossEntropyLoss',
            'grayscale': False,
            'img_key': 'img'
        }
    elif dataset_name == 'cifar100':
        return {
            'name': 'cifar100',
            'feature_type': 'img',
            'label_type': 'onehot',
            'num_classes': 100,
            'feature_transforms': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071598291397095, 0.4866936206817627, 0.44120192527770996), (0.2673342823982239, 0.2564384639263153, 0.2761504650115967))
            ]),
            'normalize': ((0.5071598291397095, 0.4866936206817627, 0.44120192527770996), (0.2673342823982239, 0.2564384639263153, 0.2761504650115967)),
            'feature_extractor': (lambda item: ([item['img'],], item['fine_label'])),
            'metric': 'OneHotMetric',
            'criterion': 'CrossEntropyLoss',
            'grayscale': False,
            'img_key': 'img'
        }
    elif dataset_name == 'imagenet-1k':
        return {
            'name': 'imagenet-1k',
            'feature_type': 'img',
            'label_type': 'onehot',
            'num_classes': 1000,
            'feature_transforms': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            'feature_extractor': (lambda item: ([item['image'],], item['label'])),
            'metric': 'OneHotMetric',
            'criterion': 'CrossEntropyLoss',
            'grayscale': False,
            'img_key': 'image'
        }