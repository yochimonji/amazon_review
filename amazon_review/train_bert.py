import copy
import random

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from util import calc_accuracy, calc_f1, init_device, load_params
from util.bert import sentence_to_loader


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
    params["batch_size"] = 8

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

    # クラスラベル設定
    for df in [train_source_df, dev_source_df, test_source_df, train_target_df, dev_target_df, test_target_df]:
        # 3以上かを予測する場合
        df["class"] = 0
        df["class"][df["stars"] > 3] = 1

        # 5クラス分類する場合
        # df["class"] = df["stars"] - 1

    # トークン化
    model_name = "cl-tohoku/bert-base-japanese-v2"
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    # dataloader作成
    train_source_dataloader = sentence_to_loader(
        train_source_df.review_body.values,
        train_source_df["class"].values,
        tokenizer,
        params["batch_size"],
        shuffle=True,
    )
    dev_source_dataloader = sentence_to_loader(
        dev_source_df.review_body.values, dev_source_df["class"].values, tokenizer, params["batch_size"], shuffle=False
    )
    # test_source_dataloader = sentence_to_loader(
    #     test_source_df.review_body.values,
    #     test_source_df["class"].values,
    #     tokenizer,
    #     params["batch_size"],
    #     shuffle=False,
    # )
    train_target_dataloader = sentence_to_loader(
        train_target_df.review_body.values,
        train_target_df["class"].values,
        tokenizer,
        params["batch_size"],
        shuffle=True,
    )
    # dev_target_dataloader = sentence_to_loader(
    #     dev_target_df.review_body.values, dev_target_df["class"].values, tokenizer, params["batch_size"], shuffle=False
    # )
    test_target_dataloader = sentence_to_loader(
        test_target_df.review_body.values,
        test_target_df["class"].values,
        tokenizer,
        params["batch_size"],
        shuffle=False,
    )

    # BERTモデル構築
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=params["class_num"],
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)

    # 最適化とスケジューラー
    # 論文で推奨されているハイパーパラメータを使用
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-6, eps=1e-8)
    epochs = 3

    # 訓練
    for epoch in range(epochs):
        print(f"\n======== Epoch {epoch+1} / {epochs} ========\nTraining")

        total_train_loss = 0
        model.train()

        for step, (input_id_batch, input_mask_batch, label_batch) in tqdm(
            enumerate(train_source_dataloader), total=len(train_source_dataloader)
        ):
            input_id_batch = input_id_batch.to(device).to(torch.int64)
            input_mask_batch = input_mask_batch.to(device).to(torch.int64)
            label_batch = label_batch.to(device).to(torch.int64)

            model.zero_grad()
            result = model(input_id_batch, token_type_ids=None, attention_mask=input_mask_batch, labels=label_batch)
            total_train_loss += result.loss.item()
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_source_dataloader)
        print(f"\n\tAverage training loss: {avg_train_loss:.2f}")

        # 検証データに対する予測
        print("\nRunning Validation")
        total_dev_loss = 0
        total_dev_accuracy = 0
        total_dev_f1 = 0
        model.eval()

        for step, (input_id_batch, input_mask_batch, label_batch) in tqdm(
            enumerate(dev_source_dataloader), total=len(dev_source_dataloader)
        ):
            input_id_batch = input_id_batch.to(device).to(torch.int64)
            input_mask_batch = input_mask_batch.to(device).to(torch.int64)
            label_batch = label_batch.to(device).to(torch.int64)

            with torch.no_grad():
                result = model(input_id_batch, token_type_ids=None, attention_mask=input_mask_batch, labels=label_batch)

            total_dev_loss += result.loss.item()
            logit_array = result.logits.detach().cpu().numpy()
            label_array = label_batch.cpu().numpy()
            total_dev_accuracy += calc_accuracy(label_array, logit_array)
            total_dev_f1 += calc_f1(label_array, logit_array)

        avg_dev_loss = total_dev_loss / len(dev_source_dataloader)
        print(f"\tDev Loss: {avg_dev_loss:.3f}")

        avg_dev_accuracy = total_dev_accuracy / len(dev_source_dataloader)
        print(f"\tAccuracy: {avg_dev_accuracy:.3f}")

        avg_dev_f1 = total_dev_f1 / len(dev_source_dataloader)
        print(f"\tF1: {avg_dev_f1:.3f}")

    # ブートストラップで複数回実行する
    print("\ntargetでFineTuning開始")
    # 事前学習したモデルを保持
    # メモリを共有しないためにdeepcopyを使用する
    model_pretrained = copy.deepcopy(model.cpu())

    params["target_ratio"] = [0.01, 0.05, 0.1, 0.3, 0.5]

    for target_ratio in params["target_ratio"]:
        print("------------------------------")
        print(f"target_ratio = {target_ratio}")
        print("------------------------------")

        accuracy_list = []
        f1_list = []

        for count in range(params["trial_count"]):
            print(f"\n{count+1}回目の試行")

            # targetでFineTuningする準備
            # target_ratioで指定した比率までtargetのデータ数を減らす
            source_num = train_source_df.shape[0]
            target_num = int(source_num * target_ratio)
            if target_num > train_target_df.shape[0]:
                print("Target ratio is too large.")
                exit()
            train_target_df_sample = train_target_df.sample(target_num, replace=False)
            print(f"Source num: {source_num}, Target num: {target_num}")

            # targetのデータローダー作成
            train_target_dataloader = sentence_to_loader(
                train_target_df_sample.review_body.values,
                train_target_df_sample["class"].values,
                tokenizer,
                params["batch_size"],
                shuffle=True,
            )

            # 事前学習したモデルをロード
            model = copy.deepcopy(model_pretrained).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=6e-6, eps=1e-8)

            # targetでFineTuning
            for epoch in range(epochs):
                print(f"======== Epoch {epoch+1} / {epochs} ========")

                total_train_loss = 0
                model.train()

                for step, (input_id_batch, input_mask_batch, label_batch) in enumerate(train_target_dataloader):
                    input_id_batch = input_id_batch.to(device).to(torch.int64)
                    input_mask_batch = input_mask_batch.to(device).to(torch.int64)
                    label_batch = label_batch.to(device).to(torch.int64)

                    model.zero_grad()
                    result = model(
                        input_id_batch, token_type_ids=None, attention_mask=input_mask_batch, labels=label_batch
                    )
                    total_train_loss += result.loss.item()
                    result.loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                avg_train_loss = total_train_loss / len(train_target_dataloader)
                print(f"Training Target Loss: {avg_train_loss:.2f}")

            # テスト
            total_test_loss = 0
            total_test_accuracy = 0
            total_test_f1 = 0
            model.eval()

            for step, (input_id_batch, input_mask_batch, label_batch) in enumerate(test_target_dataloader):
                input_id_batch = input_id_batch.to(device).to(torch.int64)
                input_mask_batch = input_mask_batch.to(device).to(torch.int64)
                label_batch = label_batch.to(device).to(torch.int64)

                with torch.no_grad():
                    result = model(
                        input_id_batch, token_type_ids=None, attention_mask=input_mask_batch, labels=label_batch
                    )

                total_test_loss += result.loss.item()
                logit_array = result.logits.detach().cpu().numpy()
                label_array = label_batch.cpu().numpy()
                total_test_accuracy += calc_accuracy(label_array, logit_array)
                total_test_f1 += calc_f1(label_array, logit_array)

            avg_test_loss = total_test_loss / len(test_target_dataloader)
            print(f"\nTest Target Loss: {avg_test_loss:.2f}")

            avg_test_accuracy = total_test_accuracy / len(test_target_dataloader)
            accuracy_list.append(avg_test_accuracy)
            print(f"Test Target Accuracy: {avg_test_accuracy:.2f}")

            avg_test_f1 = total_test_f1 / len(test_target_dataloader)
            f1_list.append(avg_test_f1)
            print(f"Test Target F1: {avg_test_f1:.2f}")

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
