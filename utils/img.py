import torch
import numpy as np
from matplotlib import pyplot as plt

def to_img(tensor, mean, std, max=255):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return ((tensor * std + mean) * 255).clamp(0, 255).permute(*range(tensor.dim() - 3), -2, -1, -3)

def show_img(imgs, filename):
    imgs = imgs.int()

    if len(imgs.shape) == 3:
        plt.imshow(imgs)
        plt.axis('off')
    elif len(imgs.shape) == 4:
        rows = min(8, imgs.shape[0])
        fig, axes = plt.subplots(1, rows, figsize=np.array([rows, 1]) * 8, gridspec_kw={"wspace": 0, "hspace": 0})

        for i in range(rows):
            axes[i].imshow(imgs[i])
            axes[i].axis('off')

    elif len(imgs.shape) == 5:
        cols = imgs.shape[0]
        rows = min(8, imgs.shape[1])

        fig, axes = plt.subplots(cols, rows, figsize=[rows * 8, cols * 8], gridspec_kw={"wspace": 0, "hspace": 0})

        if imgs.shape[1] == 1:
            for i in range(cols):
                axes[i].imshow(imgs[i, 0])
                axes[i].axis('off')
        else:
            for i in range(cols):
                for j in range(rows):
                    axes[i, j].imshow(imgs[i, j])
                    axes[i, j].axis('off')
    else:
        raise ValueError("Invalid shape for image tensor.")

    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
