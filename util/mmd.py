import torch
from torchtext.legacy import data

from util import calc_accuracy, calc_f1


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
