import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def calc_accuracy(label_list, pred_list):
    pred_flat = np.argmax(pred_list, axis=1).flatten()
    label_flat = label_list.flatten()

    return accuracy_score(label_flat, pred_flat)


def calc_f1(label_list, pred_list):
    pred_flat = np.argmax(pred_list, axis=1).flatten()
    label_flat = label_list.flatten()

    return f1_score(label_flat, pred_flat, average="macro")


def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def init_device():
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print(f"GPU available: {device}")
    else:
        device = torch.device("cpu")
        print("No GPU available, using the CPU instead.")
    return device
