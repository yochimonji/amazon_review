import copy
import random

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn
from torchtext.legacy import data
from torchtext.vocab import Vectors
from tqdm import tqdm

from util import calc_accuracy, calc_f1, init_device, load_params
from util.model import MyClassifier
from util.nlp_preprocessing import dataframe2dataset, tokenizer_ja


def main():
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
        train_df = train_df.sample(n=10000, random_state=1)
    dev_df = pd.read_json(params["ja_dev_path"], orient="record", lines=True)
    test_df = pd.read_json(params["ja_test_path"], orient="record", lines=True)

    # sourceカテゴリーとtargetカテゴリーを分ける
    train_source_df = train_df[train_df["product_category"] == params["source_category"]]
    dev_source_df = dev_df[dev_df["product_category"] == params["source_category"]]
    test_source_df = test_df[test_df["product_category"] == params["source_category"]]
    train_target_df = train_df[train_df["product_category"] == params["target_category"]]
    dev_target_df = dev_df[dev_df["product_category"] == params["target_category"]]
    test_target_df = test_df[test_df["product_category"] == params["target_category"]]

    # target_ratioで減らした際のsourceとtargetの数
    source_num = train_source_df.shape[0]
    target_num = int(source_num * params["target_ratio"])
    if target_num > train_target_df.shape[0]:
        print("Target ratio is too large.")
        exit()
    print(f"Source num: {source_num}, Target num: {target_num}")

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
    # test_source_dataset = dataframe2dataset(test_source_df, fields, columns)
    # train_target_dataset = dataframe2dataset(train_target_df, fields, columns)
    dev_target_dataset = dataframe2dataset(dev_target_df, fields, columns)
    test_target_dataset = dataframe2dataset(test_target_df, fields, columns)
    all_train_dataset = dataframe2dataset(pd.concat([train_source_df, train_target_df]), fields, columns)

    # embedding作成
    if params["use_pretrained_vector"]:
        japanese_fasttext_vectors = Vectors(name=params["ja_vector_path"])
        text_field.build_vocab(all_train_dataset, vectors=japanese_fasttext_vectors, min_freq=1)
    else:
        text_field.build_vocab(all_train_dataset, min_freq=1)

    # データローダー作成
    train_source_iter = data.BucketIterator(dataset=train_source_dataset, batch_size=params["batch_size"], train=True)
    dev_source_iter = data.BucketIterator(
        dataset=dev_source_dataset, batch_size=params["batch_size"], train=False, sort=False
    )
    # test_source_iter = data.BucketIterator(
    #     dataset=test_source_dataset, batch_size=params["batch_size"], train=False, sort=False
    # )
    # train_target_iter = data.BucketIterator(dataset=train_target_dataset, batch_size=params["batch_size"], train=True)
    dev_target_iter = data.BucketIterator(
        dataset=dev_target_dataset, batch_size=params["batch_size"], train=False, sort=False
    )
    test_target_iter = data.BucketIterator(
        dataset=test_target_dataset, batch_size=params["batch_size"], train=False, sort=False
    )

    # モデル構築
    v_size = len(text_field.vocab.stoi)
    if params["use_pretrained_vector"]:
        model = MyClassifier(params["emb_dim"], v_size, params["token_max_length"], params["class_num"], text_field).to(
            device
        )
    else:
        model = MyClassifier(params["emb_dim"], v_size, params["token_max_length"], params["class_num"]).to(device)

    criterion = getattr(nn, params["criterion"])()
    optimizer = getattr(torch.optim, params["optimizer"])(model.parameters(), lr=params["lr"])

    # sourceで訓練
    print("sourceで事前学習開始")
    for epoch in range(params["epochs"]):
        print(f"\nepoch {epoch+1} / {params['epochs']}")
        total_loss = 0

        for i, batch in tqdm(enumerate(train_source_iter), total=len(train_source_iter)):
            model.train()

            x, y = batch.text[0].to(device), batch.label.to(device)
            _, pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu()
        print(f"Train Source Loss: {total_loss / len(train_source_iter):.3f}")

        total_dev_accuracy = 0
        total_dev_f1 = 0
        model.eval()
        for valid_batch in dev_source_iter:
            x, y = valid_batch.text[0].to(device), valid_batch.label.to(device)
            with torch.no_grad():
                _, pred = model(x)
            label_array = y.cpu().numpy()
            logit_array = pred.cpu().numpy()
            total_dev_accuracy += calc_accuracy(label_array, logit_array)
            total_dev_f1 += calc_f1(label_array, logit_array)
        print(f"Dev Source Accuracy: {total_dev_accuracy / len(dev_source_iter):.2f}")
        print(f"Dev Source F1 Score: {total_dev_f1 / len(dev_source_iter):.2f}")

    # ブートストラップで複数回実行する
    print("\ntargetでFineTuning開始")
    # 事前学習したモデルを保持
    # メモリを共有しないためにdeepcopyを使用する
    model_pretrained = copy.deepcopy(model.cpu())

    accuracy_list = []
    f1_list = []

    for count in range(params["trial_count"]):
        print(f"\n{count+1}回目の試行")

        # targetでFineTuningする準備
        # target_ratioで指定した比率までtargetのデータ数を減らす
        train_target_df_sample = train_target_df.sample(target_num, replace=False)

        # targetのデータローダー作成
        train_target_dataset = dataframe2dataset(train_target_df_sample, fields, columns)
        train_target_iter = data.BucketIterator(
            dataset=train_target_dataset, batch_size=params["batch_size"], train=True
        )

        # 事前学習したモデルをロード
        model = copy.deepcopy(model_pretrained).to(device)
        optimizer = getattr(torch.optim, params["optimizer"])(model.parameters(), lr=params["lr"])

        # targetでFineTuning
        for epoch in range(params["epochs"]):
            print(f"\nepoch {epoch+1} / {params['epochs']}")
            total_loss = 0

            for i, batch in tqdm(enumerate(train_target_iter), total=len(train_target_iter)):
                model.train()

                x, y = batch.text[0].to(device), batch.label.to(device)
                _, pred = model(x)
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu()
            print(f"Train Target Loss: {total_loss / len(train_target_iter):.3f}")

            total_dev_accuracy = 0
            total_dev_f1 = 0
            model.eval()
            for valid_batch in dev_target_iter:
                x, y = valid_batch.text[0].to(device), valid_batch.label.to(device)
                with torch.no_grad():
                    _, pred = model(x)
                label_array = y.cpu().numpy()
                logit_array = pred.cpu().numpy()
                total_dev_accuracy += calc_accuracy(label_array, logit_array)
                total_dev_f1 += calc_f1(label_array, logit_array)
            print(f"Dev Target Accuracy: {total_dev_accuracy / len(dev_target_iter):.2f}")
            print(f"Dev Target F1 Score: {total_dev_f1 / len(dev_target_iter):.2f}")

        total_test_accuracy = 0
        total_test_f1 = 0
        model.eval()
        for test_batch in test_target_iter:
            x, y = test_batch.text[0].to(device), test_batch.label.to(device)
            with torch.no_grad():
                _, pred = model(x)

            label_array = y.cpu().numpy()
            logit_array = pred.cpu().numpy()
            total_test_accuracy += calc_accuracy(label_array, logit_array)
            total_test_f1 += calc_f1(label_array, logit_array)

        test_accuracy = total_test_accuracy / len(test_target_iter)
        test_f1 = total_test_f1 / len(test_target_iter)
        accuracy_list.append(test_accuracy)
        f1_list.append(test_f1)
        print(f"\nTest Target Accuracy: {test_accuracy:.2f}")
        print(f"Test Target F1 Score: {test_f1:.2f}")

    accuracy_interval = stats.t.interval(
        alpha=0.95, df=len(accuracy_list) - 1, loc=np.mean(accuracy_list), scale=stats.sem(accuracy_list)
    )
    f1_interval = stats.t.interval(alpha=0.95, df=len(f1_list) - 1, loc=np.mean(f1_list), scale=stats.sem(f1_list))
    print("\n\t\tMean, Std, 95% interval (bottom, up)")
    print(
        f"Accuracy\t{np.mean(accuracy_list):.2f}, {np.std(accuracy_list, ddof=1):.2f}, {accuracy_interval[0]:.2f}, {accuracy_interval[1]:.2f}"
    )
    print(
        f"F1 Score\t{np.mean(f1_list):.2f}, {np.std(f1_list, ddof=1):.2f}, {f1_interval[0]:.2f}, {f1_interval[1]:.2f}"
    )


if __name__ == "__main__":
    main()
