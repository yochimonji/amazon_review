import datetime

import numpy as np
from sklearn.metrics import f1_score, accuracy_score


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
