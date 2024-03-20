from abc import ABC, abstractmethod

class Defender(ABC):
    def __init__(self, def_ratio):
        self.def_ratio = def_ratio

    def prepare(self, aggregation):
        pass

    @abstractmethod
    def defense(self, grads, outputs, labels):
        pass