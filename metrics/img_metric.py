import sys
sys.path.append('..')

from metrics import Metric, OneHotMetric
import torch

class ImageMetric(Metric):
    def __init__(self):
        super().__init__()
        self.onehot_metric = OneHotMetric()

    def reset(self):
        self.loss = 0
        self.total = 0
        self.onehot_metric.reset()

    def __call__(self, pred_img, pred_label, target_img, target_label):
        self.onehot_metric(pred_label, target_label)

    def __str__(self):
        return "Loss: %.4f" % (self.loss / self.total)

    def get_log_title(self):
        return "Loss"

    def log(self):
        return str(self.loss / self.total)

def MSE(pred, target):
    return ((pred - target) ** 2).mean()

def PSNR(pred, target, max_val=255):
    mse = MSE(pred, target)
    psnr = 10 * torch.log10((max_val ** 2) / (mse + 1e-10))

    return psnr

if __name__ == "__main__":
    from utils import data, img

    dataset, data_config = data.load_dataset("cifar10", "test")
    dataloader = data.to_dataloader(dataset, 8, False)

    imgs, label = data_config['feature_extractor'](next(iter(dataloader)))
    imgs = imgs[0]

    noise = torch.randn_like(imgs) * 5
    # noise = torch.where(noise > 0.3, noise, torch.tensor(0.0))

    noise_imgs = imgs + noise
    # img = torch.stack([img, noise_img], dim=0)

    imgs = img.to_img(imgs, data_config['normalize'][0], data_config['normalize'][1])
    noise_imgs = img.to_img(noise_imgs, data_config['normalize'][0], data_config['normalize'][1])
    
    print(imgs.shape)

    PSNR(imgs, noise_imgs)

    # show_img(img, "test.png")