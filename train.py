import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup

from util import calc_accuracy, calc_f1, format_time
from util.bert import sentence_to_loader


def train():
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print("GPU available.")
    else:
        device = torch.device("cpu")
        print("No GPU available, using the CPU instead.")

    # データセット読み込み
    en_train_df = pd.read_json("./data/dataset_en_train.json", orient="record", lines=True)
    en_train_df = en_train_df.sample(n=1000, random_state=1)
    en_dev_df = pd.read_json("./data/dataset_en_dev.json", orient="record", lines=True)
    en_test_df = pd.read_json("./data/dataset_en_test.json", orient="record", lines=True)
    print("Number of en_train_df:", en_train_df.shape[0])
    print("Number of en_dev_df:", en_dev_df.shape[0])
    print("Number of en_test_df:", en_test_df.shape[0])

    # トークン化
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # dataloader作成
    batch_size = 4
    train_dataloader = sentence_to_loader(
        en_train_df.review_body.values, en_train_df.stars.values - 1, tokenizer, batch_size, shuffle=True
    )
    dev_dataloader = sentence_to_loader(
        en_dev_df.review_body.values, en_dev_df.stars.values - 1, tokenizer, batch_size, shuffle=False
    )
    # test_dataloader = sentence_to_loader(
    #     en_test_df.review_body.values, en_test_df.stars.values - 1, tokenizer, batch_size, shuffle=False
    # )

    # BERTモデル構築
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=5,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)

    # 最適化とスケジューラー
    # 論文で推奨されているハイパーパラメータを使用
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-6, eps=1e-8)
    epochs = 3
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 訓練
    training_stats = []
    total_t0 = time.time()

    for epoch in range(epochs):
        print(f"\n======== Epoch {epoch+1} / {epochs} ========\nTraining")

        total_train_loss = 0
        model.train()

        for step, (input_id_batch, input_mask_batch, label_batch) in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
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
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.3f}")

        # 検証データに対する予測
        print("\nRunning Validation")
        total_dev_loss = 0
        total_dev_accuracy = 0
        total_dev_f1 = 0
        model.eval()

        for step, (input_id_batch, input_mask_batch, label_batch) in tqdm(
            enumerate(dev_dataloader), total=len(dev_dataloader)
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

        avg_dev_loss = total_dev_loss / len(dev_dataloader)
        print(f"Dev Loss: {avg_dev_loss:.3f}")

        avg_dev_accuracy = total_dev_accuracy / len(dev_dataloader)
        print(f"Accuracy: {avg_dev_accuracy:.3f}")

        avg_dev_f1 = total_dev_f1 / len(dev_dataloader)
        print(f"F1: {avg_dev_f1:.3f}")

        training_stats.append(
            {
                "epoch": epoch + 1,
                "Training Loss": avg_train_loss,
                "Dev Loss": avg_dev_loss,
                "Dev Accuracy": avg_dev_accuracy,
                "Dev F1": avg_dev_f1,
            }
        )

    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time() - total_t0)} (h:mm:ss)")


if __name__ == "__main__":
    train()
