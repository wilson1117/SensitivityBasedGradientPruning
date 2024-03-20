from argparse import ArgumentParser
from functools import partial
import utils
import torch
from torch import optim
import random
import defender
import os
import aggregation
from models import get_classification_model

DATA_CONFIG_PATH = "data_config"

if __name__ == '__main__':
    parser = ArgumentParser()

    #root
    parser.add_argument("--root-dir", type=str, default="result", help="Root directory for data.")

    #dataset config
    parser.add_argument("--data-config", type=str, default="mnist_100_2_42.json", help="Dataset configuration file.")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for dataloader.")
    parser.add_argument("--test-dataset", type=str, default=None, help="Test dataset for federated learning.")
    parser.add_argument("--test-batch", type=int, default=128, help="Batch size for test dataloader.")

    #model config
    parser.add_argument("--model", type=str, default="LeNet", help="Model for federated learning.")
    parser.add_argument("--pretrain", type=str, default=None, help="Pretrained model for federated learning.")

    #train config
    parser.add_argument("--criterion", type=str, default=None, help="Criterion for federated learning.")
    parser.add_argument("--metric", type=str, default=None, help="Metric for federated learning.")
    parser.add_argument("--device", type=str, default=None, help="Device for federated learning.")
    parser.add_argument("--save-freq", type=int, default=10, help="Save frequency for trained models.")

    #federated config
    parser.add_argument("--agg", type=str, default="FedSGD", help="Aggregation method for federated learning.")
    parser.add_argument("--rounds", type=int, default=500, help="Number of rounds for federated learning.")
    parser.add_argument("--random-select", type=int, default=50, help="Number of random selected clients for federated learning.")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer for federated learning.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for federated optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for federated optimizer.")

    #other config
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory for federated learning.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for federated learning.")

    #defender config
    parser.add_argument("--defender", type=str, default=None, help="Defender for federated learning.")
    parser.add_argument("--def-ratio", type=float, default=0.5, help="Ratio for defender.")

    args = parser.parse_args()

    # Set Random Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    #load dataset
    (pub_set, val_set, train_set), dataset_config = utils.data.load_subsets(os.path.join(DATA_CONFIG_PATH, args.data_config))

    if args.test_dataset is None:
        args.test_dataset = dataset_config['name']

    test_dataset, test_dataset_config = utils.data.load_dataset(args.test_dataset, "test")
    train_loader = utils.data.to_dataloader(train_set, args.batch_size, shuffle=True)
    pub_loader = utils.data.to_dataloader(pub_set, len(pub_set), shuffle=False)
    val_loader = utils.data.to_dataloader(val_set, args.test_batch, shuffle=False)
    test_loader = utils.data.to_dataloader(test_dataset, args.test_batch, shuffle=False)

    #load model
    model = get_classification_model(args.model, dataset_config['num_classes'], dataset_config['label_type'], args.pretrain)

    #load defender
    if args.defender is not None:
        def_obj = getattr(defender, args.defender)(args.def_ratio)
    else:
        def_obj = None

    if args.optimizer == "SGD":
        optimizer = partial(optim.SGD, lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "Adam":
        optimizer = partial(optim.Adam, lr=args.lr)

    if args.log_dir is None:
        args.log_dir = f"{args.model}_{args.agg}_{dataset_config['name']}_{args.defender if args.defender is not None else 'Origin'}_{len(train_loader)}_{args.batch_size}"

    logger = utils.logger.Logger(os.path.join(args.root_dir, args.log_dir))

    logger.create_log("params.json").write_json(vars(args))

    agg_obj = getattr(aggregation, args.agg)(model, dataset_config, optimizer, logger, pub_loader, def_obj,
                                             args.metric, args.criterion, val_loader, test_loader, args.device)


    agg_obj.run(train_loader, args.rounds, args.random_select, args.save_freq)