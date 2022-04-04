import torch
from torchtext.legacy import data

from util import calc_accuracy, calc_f1


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


def run_test(model: torch.nn.Module, iter: data.BucketIterator, device: torch.device):
    total_accuracy = 0
    total_f1 = 0
    model.eval()
    for batch in iter:
        x, y = batch.text[0].to(device), (batch.label - 1).to(device)
        with torch.no_grad():
            _, pred = model(x)

        label_array = y.cpu().numpy()
        logit_array = pred.cpu().numpy()

        total_accuracy += calc_accuracy(label_array, logit_array)
        total_f1 += calc_f1(label_array, logit_array)
        avg_accuracy = total_accuracy / len(iter)
        avg_f1 = total_f1 / len(iter)
    return avg_accuracy, avg_f1
