import sys
sys.path.append("..")

import os
from .Aggregation import Aggregation
import torch

class FedSGD(Aggregation):
    def __init__(self, central_model, data_config, optimizer, logger, public_loader,
                 defender=None, metric=None, criterion=None, val_loader=None, test_loader=None, device=None):
        
        super(FedSGD, self).__init__(central_model, data_config, optimizer, logger, public_loader, defender, metric, criterion, val_loader, test_loader, device)

    def prepare(self):
        if self.defender is not None:
            self.defender.prepare(self)

    def train(self, data):
        features, labels = self.feature_extractor(data)
        features = [ feature.to(self.device) for feature in features ]
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.central_model(*features)
        loss = self.criterion(outputs, labels)

        self.loss_metric(loss.item(), len(labels))
        self.metric(outputs, labels)

        grad = torch.autograd.grad(loss, self.central_model.parameters())

        if self.defender is not None:
            grad = self.defender.defense(grad, outputs, labels)

        return [g.cpu() for g in grad]

    def aggregation(self, grads, select_num):
        self.optimizer.zero_grad()

        for idx, param in enumerate(self.central_model.parameters()):
            param.grad = torch.sum(torch.stack([grad[idx] for grad in grads]), dim=0).to(self.device) / select_num

        self.optimizer.step()