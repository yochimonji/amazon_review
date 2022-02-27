from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader


def tokenize_map(sentence, tokenizer: BertTokenizer, max_length=512):
    input_id_list = []
    attention_mask_list = []

    for text in sentence:
        # tokenizer.encode_plus() は非推奨だけどとりあえず参考にしたnotebook通りに実装
        # tokenizer() が推奨
        encoded_dict = tokenizer(
            text,
            add_special_tokens=True,
            truncation="longest_first",
            pad_to_max_length=True,  # 非推奨 padding=True に変える
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_id_list.append(encoded_dict["input_ids"])
        attention_mask_list.append(encoded_dict["attention_mask"])

    input_id_list = torch.cat(input_id_list, dim=0)
    attention_mask_list = torch.cat(attention_mask_list, dim=0)

    return input_id_list, attention_mask_list


def sentence_to_loader(sentence, label_list, tokenizer: BertTokenizer, batch_size=32, shuffle=False):
    input_id_tensor, attention_tensor = tokenize_map(sentence, tokenizer)
    label_tensor = torch.tensor(label_list)
    dataset = TensorDataset(input_id_tensor, attention_tensor, label_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
