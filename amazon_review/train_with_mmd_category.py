import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from torch import nn
from torchtext.legacy import data
from torchtext.vocab import Vectors
from tqdm import tqdm

from util import init_device, load_params
from util.mmd import run_test
from util.model import MMD, MyEmbedding, MyMLP
from util.nlp_preprocessing import dataframe2dataset, tokenizer_ja


def train():
    # ランダムシード初期化
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = init_device()

    # パラメータ読み込み
    print("Loading parameters...")
    params = load_params("/workspace/amazon_review/config/params_mmd.json")

    # データセット読み込み
    train_df = pd.read_json(params["ja_train_path"], orient="record", lines=True)
    if params["is_developing"]:
        train_df = train_df.sample(n=200000, random_state=1)
    dev_df = pd.read_json(params["ja_dev_path"], orient="record", lines=True)
    test_df = pd.read_json(params["ja_test_path"], orient="record", lines=True)

    # sourceカテゴリーとtargetカテゴリーを分ける
    train_source_df = train_df[train_df["product_category"] == params["source_category"]]
    dev_source_df = dev_df[dev_df["product_category"] == params["source_category"]]
    test_source_df = test_df[test_df["product_category"] == params["source_category"]]
    train_target_df = train_df[train_df["product_category"] == params["target_category"]]
    dev_target_df = dev_df[dev_df["product_category"] == params["target_category"]]
    test_target_df = test_df[test_df["product_category"] == params["target_category"]]

    # target_ratioで指定した比率までtargetのデータ数を減らす
    source_num = train_source_df.shape[0]
    target_num = int(source_num * params["target_ratio"])
    if target_num > train_target_df.shape[0]:
        print("Target ratio is too large...")
        print(f"Source num: {train_source_df.shape[0]}, Target num: {train_target_df.shape[0]}")
        exit()
    train_target_df = train_target_df.sample(target_num, replace=False)
    print(f"Source num: {train_source_df.shape[0]}, Target num: {train_target_df.shape[0]}")

    # クラスラベル設定
    for df in [train_source_df, dev_source_df, test_source_df, train_target_df, dev_target_df, test_target_df]:
        # 3以上かを予測する場合
        df["class"] = 0
        df["class"][df["stars"] > 3] = 1

        # 5クラス分類する場合
        # df["class"] = df["stars"] - 1

    # フィールド作成
    print("Building data iterator...")
    text_field = data.Field(
        sequential=True,
        tokenize=tokenizer_ja,
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

    # データセット作成
    columns = ["review_body", "class"]
    train_source_dataset = dataframe2dataset(train_source_df, fields, columns)
    dev_source_dataset = dataframe2dataset(dev_source_df, fields, columns)
    test_source_dataset = dataframe2dataset(test_source_df, fields, columns)
    train_target_dataset = dataframe2dataset(train_target_df, fields, columns)
    dev_target_dataset = dataframe2dataset(dev_target_df, fields, columns)
    test_target_dataset = dataframe2dataset(test_target_df, fields, columns)
    all_train_dataset = dataframe2dataset(pd.concat([train_source_df, train_target_df]), fields, columns)

    # embedding作成
    if params["use_pretrained_vector"]:
        japanese_fasttext_vectors = Vectors(name=params["ja_vector_path"])
        text_field.build_vocab(all_train_dataset, vectors=japanese_fasttext_vectors, min_freq=1)
    else:
        text_field.build_vocab(all_train_dataset, min_freq=1)

    # データセット作成
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

    # モデル構築
    v_size = len(text_field.vocab.stoi)
    if params["use_pretrained_vector"]:
        source_embedding = MyEmbedding(
            params["emb_dim"], v_size, params["token_max_length"], params["class_num"], text_field
        ).to(device)
        target_embedding = MyEmbedding(
            params["emb_dim"], v_size, params["token_max_length"], params["class_num"], text_field
        ).to(device)
    else:
        source_embedding = MyEmbedding(params["emb_dim"], v_size, params["token_max_length"]).to(device)
        target_embedding = MyEmbedding(params["emb_dim"], v_size, params["token_max_length"]).to(device)
    mlp = MyMLP(class_num=params["class_num"]).to(device)
    criterion = getattr(nn, params["criterion"])()
    mmd = MMD("rbf")

    optimizer_s_emb = getattr(torch.optim, params["optimizer"])(source_embedding.parameters(), lr=params["lr"])
    optimizer_t_emb = getattr(torch.optim, params["optimizer"])(target_embedding.parameters(), lr=params["lr"])
    optimizer_mlp = getattr(torch.optim, params["optimizer"])(mlp.parameters(), lr=params["lr"])

    split_ratio = len(train_target_dataset) / len(train_source_dataset)

    # 訓練
    for epoch in range(params["epochs"]):
        print(f"\nepoch {epoch+1} / {params['epochs']}")
        # 各lossの初期化
        total_source_loss = 0
        total_target_loss = 0
        total_mmd_loss = 0
        total_all_loss = 0

        # MMDのためにターゲットと同数のソースが必要なためエポックごとにソースのデータローダーを作成し直す
        # エポックごとに異なるソースのデータが使用されるようになる
        random.seed(epoch)
        train_source_subset, _ = train_source_dataset.split(split_ratio=split_ratio)
        train_source_iter = data.BucketIterator(
            dataset=train_source_subset, batch_size=params["batch_size"], train=True
        )

        for i, (source_batch, target_batch) in tqdm(
            enumerate(zip(train_source_iter, train_target_iter)), total=len(train_source_iter)
        ):
            source_embedding.train()
            target_embedding.train()
            mlp.train()

            source_x, source_y = source_batch.text[0].to(device), (source_batch.label).to(device)
            target_x, target_y = target_batch.text[0].to(device), (target_batch.label).to(device)

            # MMDの処理はBatch数が同数でなけらばならないためcontinueする
            # sourceとtargetで毎回同じBatch数だけデータがロードされる処理ができれば下の処理は不要
            if source_x.shape[0] != params["batch_size"] or target_x.shape[0] != params["batch_size"]:
                continue

            source_embed = source_embedding(source_x)
            source_pred = mlp(source_embed)
            source_loss = criterion(source_pred, source_y)
            total_source_loss += source_loss.cpu()

            target_embed = target_embedding(target_x)
            target_pred = mlp(target_embed)
            target_loss = criterion(target_pred, target_y)
            total_target_loss += target_loss.cpu()

            if params["lambda"] == 0:
                all_loss = source_loss + target_loss
            else:
                mmd_loss = mmd(source_embed, target_embed)
                total_mmd_loss += mmd_loss.cpu()
                all_loss = source_loss + target_loss + params["lambda"] * mmd_loss

            optimizer_s_emb.zero_grad()
            optimizer_t_emb.zero_grad()
            optimizer_mlp.zero_grad()

            all_loss.backward()

            optimizer_s_emb.step()
            optimizer_t_emb.step()
            optimizer_mlp.step()
            total_all_loss += all_loss.cpu()

        mean_source_loss = total_source_loss / len(train_source_iter)
        mean_target_loss = total_target_loss / len(train_target_iter)
        mean_all_loss = total_all_loss / len(train_source_iter)
        if params["lambda"] == 0:
            print(f"Loss -> Source: {mean_source_loss:.3f}\tTarget: {mean_target_loss:.3f}\tAll: {mean_all_loss:.3f}")
        else:
            mean_mmd_loss = total_mmd_loss / len(train_source_iter)
            print(
                f"Loss -> Source: {mean_source_loss:.3f}\tTarget: {mean_target_loss:.3f}\tMMD: {mean_mmd_loss:.3f}\tAll: {mean_all_loss:.3f}"  # noqa #E501
            )

        dev_source_accuracy, dev_source_f1 = run_test(source_embedding, mlp, dev_source_iter, device)
        print(f"\nDev source Accuracy: {dev_source_accuracy:.2f}")
        print(f"Dev source F1 Score: {dev_source_f1:.2f}")
        dev_target_accuracy, dev_target_f1 = run_test(target_embedding, mlp, dev_target_iter, device)
        print(f"\nDev target Accuracy: {dev_target_accuracy:.2f}")
        print(f"Dev target F1 Score: {dev_target_f1:.2f}")

    test_source_accuracy, test_source_f1 = run_test(source_embedding, mlp, test_source_iter, device)
    print(f"\nTest source Accuracy: {test_source_accuracy:.2f}")
    print(f"Test source F1 Score: {test_source_f1:.2f}")
    test_target_accuracy, test_target_f1 = run_test(target_embedding, mlp, test_target_iter, device)
    print(f"\nTest target Accuracy: {test_target_accuracy:.2f}")
    print(f"Test target F1 Score: {test_target_f1:.2f}")

    # 特徴量可視化
    source_embedding.eval()
    target_embedding.eval()

    train_source_iter = data.BucketIterator(dataset=train_source_dataset, batch_size=params["batch_size"], train=True)
    source_embedding_list = []
    for batch in train_source_iter:
        x, _ = batch.text[0].to(device), (batch.label).to(device)
        with torch.no_grad():
            embedding = source_embedding(x)
        source_embedding_list.extend(embedding.cpu().numpy())
    source_df = pd.DataFrame(np.array(source_embedding_list))

    target_embedding_list = []
    for batch in train_target_iter:
        x, _ = batch.text[0].to(device), (batch.label).to(device)
        with torch.no_grad():
            embedding = target_embedding(x)
        target_embedding_list.extend(embedding.cpu().numpy())
    target_df = pd.DataFrame(np.array(target_embedding_list))

    pca = PCA(n_components=2)
    pca.fit(source_df)
    source_pca_df = pca.transform(source_df)
    target_pca_df = pca.transform(target_df)
    source_pca_df.shape

    plt.scatter(source_pca_df[:, 0], source_pca_df[:, 1], label="Source")
    plt.scatter(target_pca_df[:, 0], target_pca_df[:, 1], label="Target")
    plt.legend()
    plt.show()

    label_list = []
    pred_list = []
    for batch in test_target_iter:
        x, y = batch.text[0].to(device), (batch.label).to(device)
        with torch.no_grad():
            embedding = target_embedding(x)
            pred = mlp(embedding)
        label_list.extend(list(y.cpu().numpy()))
        pred_list.extend(list(pred.cpu().numpy().argmax(1)))
    print(confusion_matrix(label_list, pred_list))


if __name__ == "__main__":
    train()
