import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchtext.legacy import data
from torchtext.vocab import Vectors
from tqdm import tqdm

from util import calc_accuracy, calc_f1, init_device, load_params
from util.model import MyClassifier
from util.nlp_preprocessing import dataframe2dataset, tokenizer


def train():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Loading parameters...")
    params = load_params("config/params_domain_shift.json")

    device = init_device()

    # データセット読み込み
    ja_train_df = pd.read_json(params["train_data_path"], orient="record", lines=True)
    if params["is_developing"]:
        ja_train_df = ja_train_df.sample(n=10000, random_state=1)
    ja_dev_df = pd.read_json(params["dev_data_path"], orient="record", lines=True)
    ja_test_df = pd.read_json(params["test_data_path"], orient="record", lines=True)

    train_df = ja_train_df[ja_train_df["product_category"].isin(params["use_category_list"])]
    dev_df = ja_dev_df[ja_dev_df["product_category"].isin(params["use_category_list"])]
    test_df = ja_test_df[ja_test_df["product_category"].isin(params["use_category_list"])]
    print("Numer of train:", train_df.shape[0])
    print("Numer of dev:", dev_df.shape[0])
    print("Numer of test:", test_df.shape[0])

    print("Building data iterator...")
    text_field = data.Field(
        sequential=True,
        tokenize=tokenizer,
        use_vocab=True,
        lower=True,
        include_lengths=True,
        batch_first=True,
        fix_length=params["token_max_length"],
        init_token="<cls>",
        eos_token="<eos>",
    )
    label_field = data.Field(
        sequential=False, use_vocab=False, preprocessing=lambda s: params["use_category_list"].index(s)
    )
    fields = [("text", text_field), ("label", label_field)]

    columns = ["review_body", "product_category"]
    train_dataset = dataframe2dataset(train_df, fields, columns)
    dev_dataset = dataframe2dataset(dev_df, fields, columns)
    test_dataset = dataframe2dataset(test_df, fields, columns)

    japanese_fasttext_vectors = Vectors(name=params["vector_path"])
    text_field.build_vocab(train_dataset, vectors=japanese_fasttext_vectors, min_freq=1)

    train_iter = data.BucketIterator(dataset=train_dataset, batch_size=params["batch_size"], train=True)
    dev_iter = data.BucketIterator(dataset=dev_dataset, batch_size=params["batch_size"], train=False, sort=False)
    test_iter = data.BucketIterator(dataset=test_dataset, batch_size=params["batch_size"], train=False, sort=False)

    v_size = len(text_field.vocab.stoi)
    model = MyClassifier(
        params["emb_dim"], v_size, params["token_max_length"], len(params["use_category_list"]), text_field
    ).to(device)

    criterion = getattr(nn, params["criterion"])()
    optimizer = getattr(torch.optim, params["optimizer"])(model.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(params["epochs"]):
        print(f"\nepoch {epoch+1} / {params['epochs']}")

        total_loss = 0
        for i, batch in tqdm(enumerate(train_iter), total=len(train_iter)):
            model.train()
            optimizer.zero_grad()
            x, y = batch.text[0].to(device), batch.label.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.cpu()
        scheduler.step()
        print(f"Train Loss: {total_loss / len(train_iter):.3f}")

        total_dev_accuracy = 0
        total_dev_f1 = 0
        model.eval()
        for valid_batch in dev_iter:
            x, y = valid_batch.text[0].to(device), valid_batch.label.to(device)
            with torch.no_grad():
                pred = model(x)
            label_array = y.cpu().numpy()
            logit_array = pred.cpu().numpy()
            total_dev_accuracy += calc_accuracy(label_array, logit_array)
            total_dev_f1 += calc_f1(label_array, logit_array)
        print(f"Dev Accuracy: {total_dev_accuracy / len(dev_iter):.2f}")
        print(f"Dev F1 Score: {total_dev_f1 / len(dev_iter):.2f}")

    total_test_accuracy = 0
    total_test_f1 = 0
    model.eval()
    for test_batch in test_iter:
        x, y = test_batch.text[0].to(device), test_batch.label.to(device)
        with torch.no_grad():
            pred = model(x)

        label_array = y.cpu().numpy()
        logit_array = pred.cpu().numpy()
        total_test_accuracy += calc_accuracy(label_array, logit_array)
        total_test_f1 += calc_f1(label_array, logit_array)
    print(f"\nTest Accuracy: {total_test_accuracy / len(test_iter):.2f}")
    print(f"Test F1 Score: {total_test_f1 / len(test_iter):.2f}")


if __name__ == "__main__":
    train()
