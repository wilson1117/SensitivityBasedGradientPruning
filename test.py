from utils import *

(pub_set, val_set, train_set), dataset_config = data.load_subsets('data_config/mnist_100_2_42.json')

print(val_set)