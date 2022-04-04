import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchtext.legacy import data
from torchtext.vocab import Vectors
from tqdm import tqdm

from util import init_device, load_params
from util.mmd import MMD, run_test
from util.model import MLPWithMMD
from util.nlp_preprocessing import dataframe2dataset, tokenizer


def train():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Loading parameters...")
    params = load_params("config/params_mmd.json")

    device = init_device()

    # データセット読み込み
    train_df = pd.read_json(params["train_data_path"], orient="record", lines=True)
    if params["is_developing"]:
        train_df = train_df.sample(n=5000, random_state=1)
    dev_df = pd.read_json(params["dev_data_path"], orient="record", lines=True)
    test_df = pd.read_json(params["test_data_path"], orient="record", lines=True)

    train_source_df = train_df[train_df["product_category"] == params["source_category"]]
    dev_source_df = dev_df[dev_df["product_category"] == params["source_category"]]
    test_source_df = test_df[test_df["product_category"] == params["source_category"]]
    train_target_df = train_df[train_df["product_category"] == params["target_category"]]
    dev_target_df = dev_df[dev_df["product_category"] == params["target_category"]]
    test_target_df = test_df[test_df["product_category"] == params["target_category"]]

    if train_source_df.shape[0] < train_target_df.shape[0]:
        print("Target more than source")
        print(f"Source num: {train_source_df.shape[0]}, Target num: {train_target_df.shape[0]}")
        exit()
    train_target_num = int(min(train_source_df.shape[0] * params["target_ratio"], train_target_df.shape[0]))
    train_target_df = train_target_df.sample(train_target_num, replace=False)
    print(f"Source num: {train_source_df.shape[0]}, Target num: {train_target_df.shape[0]}")

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
    label_field = data.Field(sequential=False, use_vocab=False)
    fields = [("text", text_field), ("label", label_field)]

    columns = ["review_body", "stars"]
    train_source_dataset = dataframe2dataset(train_source_df, fields, columns)
    dev_source_dataset = dataframe2dataset(dev_source_df, fields, columns)
    test_source_dataset = dataframe2dataset(test_source_df, fields, columns)
    train_target_dataset = dataframe2dataset(train_target_df, fields, columns)
    dev_target_dataset = dataframe2dataset(dev_target_df, fields, columns)
    test_target_dataset = dataframe2dataset(test_target_df, fields, columns)

    japanese_fasttext_vectors = Vectors(name=params["vector_path"])
    text_field.build_vocab(train_source_dataset, vectors=japanese_fasttext_vectors, min_freq=1)

    # train_source_iterのみエポックごとに生成し直す
    dev_source_iter = data.BucketIterator(
        dataset=dev_source_dataset, batch_size=params["batch_size"], train=False, sort=False
    )
    test_source_iter = data.BucketIterator(
        dataset=test_source_dataset, batch_size=params["batch_size"], train=False, sort=False
    )
    train_target_iter = data.BucketIterator(dataset=train_target_dataset, batch_size=params["batch_size"], train=True)
    dev_target_iter = data.BucketIterator(
        dataset=dev_target_dataset, batch_size=params["batch_size"], train=False, sort=False
    )
    test_target_iter = data.BucketIterator(
        dataset=test_target_dataset, batch_size=params["batch_size"], train=False, sort=False
    )

    v_size = len(text_field.vocab.stoi)
    model = MLPWithMMD(params["emb_dim"], v_size, params["token_max_length"], params["class_num"], text_field).to(
        device
    )
    criterion = getattr(nn, params["criterion"])()
    optimizer = getattr(torch.optim, params["optimizer"])(model.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    split_ratio = len(train_target_dataset) / len(train_source_dataset)

    for epoch in range(params["epochs"]):
        print(f"\nepoch {epoch+1} / {params['epochs']}")
        total_loss = 0

        random.seed(epoch)
        train_source_subset, _ = train_source_dataset.split(split_ratio=split_ratio)
        train_source_iter = data.BucketIterator(
            dataset=train_source_subset, batch_size=params["batch_size"], train=True
        )

        for i, (source_batch, target_batch) in tqdm(
            enumerate(zip(train_source_iter, train_target_iter)), total=len(train_source_iter)
        ):
            model.train()
            optimizer.zero_grad()
            source_x, source_y = source_batch.text[0].to(device), (source_batch.label - 1).to(device)
            target_x, target_y = target_batch.text[0].to(device), (target_batch.label - 1).to(device)

            if source_x.shape[0] != params["batch_size"] or target_x.shape[0] != params["batch_size"]:
                continue

            source_embed, source_pred = model(source_x)
            source_loss = criterion(source_pred, source_y)

            target_embed, target_pred = model(target_x)
            target_loss = criterion(target_pred, target_y)

            if params["lambda"] == 0:
                all_loss = source_loss + target_loss
                all_loss.backward()
                # print(source_loss.cpu().item(), target_loss.cpu().item(), all_loss.cpu().item())
            else:
                mmd_loss = MMD(source_embed, target_embed, "multiscale", device)
                all_loss = source_loss + target_loss + params["lambda"] * mmd_loss
                all_loss.backward()
                # print(source_loss.cpu().item(), target_loss.cpu().item(), mmd_loss.cpu().item(), all_loss.cpu().item())

            optimizer.step()
            total_loss += all_loss.cpu()

        scheduler.step()
        print(f"Train Loss: {total_loss / (len(train_source_iter) + len(train_target_iter)):.3f}")

        dev_source_accuracy, dev_source_f1 = run_test(model, dev_source_iter, device)
        print(f"\nDev source Accuracy: {dev_source_accuracy:.2f}")
        print(f"Dev source F1 Score: {dev_source_f1:.2f}")
        dev_target_accuracy, dev_target_f1 = run_test(model, dev_target_iter, device)
        print(f"\nDev target Accuracy: {dev_target_accuracy:.2f}")
        print(f"Dev target F1 Score: {dev_target_f1:.2f}")

    test_source_accuracy, test_source_f1 = run_test(model, test_source_iter, device)
    print(f"\nTest source Accuracy: {test_source_accuracy:.2f}")
    print(f"Test source F1 Score: {test_source_f1:.2f}")
    test_target_accuracy, test_target_f1 = run_test(model, test_target_iter, device)
    print(f"\nTest target Accuracy: {test_target_accuracy:.2f}")
    print(f"Test target F1 Score: {test_target_f1:.2f}")


if __name__ == "__main__":
    train()
