import sys
sys.path.append("..")

from abc import ABC, abstractmethod
import torch
from metrics import get_metric, get_logs_title, get_logs, dump_logs, LossMetric
from tqdm import tqdm, trange
import random
import os

class Aggregation(ABC):
    def __init__(self, central_model, data_config, optimizer, logger, public_loader,
                 defender=None, metric=None, criterion=None, val_loader=None, test_loader=None, device=None):
        
        self.loss_metric = LossMetric()
        
        self.central_model = central_model
        self.data_config = data_config
        self.feature_extractor = data_config['feature_extractor']
        self.public_loader = self.feature_extractor(next(iter(public_loader))) if public_loader is not None else None
        self.optimizer = optimizer(self.central_model.parameters())
        self.logger = logger
        self.defender = defender
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.metric = get_metric(data_config['metric'])() if metric is None else get_metric(metric)()
        self.criterion = getattr(torch.nn, data_config['criterion'])() if criterion is None else getattr(torch.nn, criterion)()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    
    def run(self, client_loaders, rounds, random_select, save_freq):

        train_logger = self.logger.create_log("train.csv")

        if self.val_loader is not None:
            train_logger.log("Round, Train Loss, Train Acc, Val Loss, Val Acc")
        else:
            train_logger.log("Round, Train Loss, Train Acc")

        client_iter = [[idx, iter(loader)] for idx, loader in enumerate(client_loaders)]

        model_folder = self.logger.create_folder("models")

        self.move_central_model(self.device)

        for r in range(rounds):
            print("\nRound", r + 1)
            self.prepare()

            random.shuffle(client_iter)
            self.central_model.train()
            
            self.loss_metric.reset()
            self.metric.reset()

            grads = []

            for idx in trange(random_select):
                try:
                    data = next(client_iter[idx][1])
                except StopIteration:
                    client_iter[idx][1] = iter(client_loaders[client_iter[idx][0]])
                    data = next(client_iter[idx][1])

                grad = self.train(data)

                grads.append(grad)

            self.aggregation(grads, random_select)

            dump_logs(self.loss_metric, self.metric)

            if self.val_loader is not None:
                print("Evaling")
                train_logger.log(f"{r + 1}, {get_logs(self.loss_metric, self.metric)}, {self.eval(self.val_loader)}")

            if r == 0 or (r + 1) % save_freq == 0:
                torch.save(self.central_model.state_dict(), os.path.join(model_folder, "Round_" + str(r + 1) + ".pth"))
        
        if self.test_loader is not None:
            print("\nTest")
            result = self.eval(self.test_loader)

            test_logger = self.logger.create_log("test.csv")
            test_logger.log(get_logs_title(self.loss_metric, self.metric))
            test_logger.log(result)

        self.move_central_model("cpu")


    def eval(self, dataloader):
        self.central_model.eval()
        self.loss_metric.reset()
        self.metric.reset()

        with torch.no_grad():
            for data in tqdm(dataloader):
                features, labels = self.feature_extractor(data)
                features = [ feature.to(self.current_device) for feature in features ]
                labels = labels.to(self.current_device)

                outputs = self.central_model(*features)
                loss = self.criterion(outputs, labels)

                self.loss_metric(loss.item(), len(labels))
                self.metric(outputs, labels)

        dump_logs(self.loss_metric, self.metric)
        
        return get_logs(self.loss_metric, self.metric)

    def move_central_model(self, device):
        self.central_model.to(device)
        self.current_device = device
    
    def prepare(self):
        pass

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def aggregation(self, grads, select_num):
        pass