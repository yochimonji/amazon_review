import pandas as pd
import torch
import torchtext
from janome.tokenizer import Tokenizer
from torch import nn
from torchtext.legacy import data
from torchtext.vocab import FastText
from tqdm import tqdm

from util import init_device


def train():
    device = init_device()

    # データセット読み込み
    ja_train_df = pd.read_json("./data/dataset_ja_train.json", orient="record", lines=True)
    ja_dev_df = pd.read_json("./data/dataset_ja_dev.json", orient="record", lines=True)
    ja_test_df = pd.read_json("./data/dataset_ja_test.json", orient="record", lines=True)

    ja_train_home_df = ja_train_df[ja_train_df["product_category"] == "home"]
    ja_dev_home_df = ja_dev_df[ja_dev_df["product_category"] == "home"]
    ja_test_home_df = ja_test_df[ja_test_df["product_category"] == "home"]
    ja_train_jewelry_df = ja_train_df[ja_train_df["product_category"] == "jewelry"]
    ja_dev_jewelry_df = ja_dev_df[ja_dev_df["product_category"] == "jewelry"]
    ja_test_jewelry_df = ja_test_df[ja_test_df["product_category"] == "jewelry"]
    print(
        f"Number of home (train, dev, test) = ({ja_train_home_df.shape[0]}, {ja_dev_home_df.shape[0]}, {ja_test_home_df.shape[0]})"
    )
    print(
        f"Number of jewelry (train, dev, test) = ({ja_train_jewelry_df.shape[0]}, {ja_dev_jewelry_df.shape[0]}, {ja_test_jewelry_df.shape[0]})"
    )


if __name__ == "__main__":
    train()
