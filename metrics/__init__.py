from .metric import Metric, get_logs_title, get_logs, dump_logs
from .onehot_metric import OneHotMetric
from .loss_metric import LossMetric
from .img_metric import ImageMetric

__all__ = ['Metric', 'OneHotMetric', 'LossMetric', 'get_logs_title', 'get_logs', 'dump_logs', 'ImageMetric']

metric_list = {
    'OneHotMetric': OneHotMetric,
    'LossMetric': LossMetric,
    'ImageMetric': ImageMetric
}

def get_metric(name):
    return metric_list[name]