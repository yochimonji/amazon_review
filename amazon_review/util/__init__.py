import datetime
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def calc_accuracy(label_list, pred_list):
    pred_flat = np.argmax(pred_list, axis=1).flatten()
    label_flat = label_list.flatten()

    return accuracy_score(label_flat, pred_flat) * 100


def calc_f1(label_list, pred_list):
    pred_flat = np.argmax(pred_list, axis=1).flatten()
    label_flat = label_list.flatten()

    return f1_score(label_flat, pred_flat, average="macro") * 100


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


def print_params(params, nest=0):
    for param in params:
        print("\t" * nest, param, end=":")
        if type(params[param]) == dict:
            print("{")
            print_params(params[param], nest=nest + 1)
            print("}\n")
        else:
            print("\t", params[param])


# jsonファイルを読み込んでパラメータを設定する
# jsonから読み込むことでpyファイルの書き換えをしなくてよいのでGitが汚れない
def load_params(path="/amazon_review/config/params.json"):
    if len(sys.argv) == 2:
        if os.path.exists(sys.argv[1]):
            path = sys.argv[1]
        else:
            print("Error:指定した引数のパスにファイルが存在しません")
            sys.exit()
    with open(path, "r") as file:
        params = json.load(file)
    print_params(params)
    return params
