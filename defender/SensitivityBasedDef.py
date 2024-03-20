from .Defender import Defender

class SensitivityBasedDefender(Defender):
    def __init__(self, def_ratio):
        super(SensitivityBasedDefender, self).__init__(def_ratio)

    def prepare(self, aggregation):
        pass

    def defense(self, grads, outputs, labels):
        pass
        