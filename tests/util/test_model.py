import pandas as pd
import torch
from amazon_review.util.model import MMD, MyClassifier
from amazon_review.util.nlp_preprocessing import dataframe2dataset, tokenizer
from torchtext.legacy import data
from torchtext.vocab import Vectors


def test_myclassifier_1() -> None:
    batch_size = 24
    emb_dim = 300
    v_size = 100
    max_length = 256
    class_num = 5

    model = MyClassifier(emb_dim, v_size, max_length, class_num, None)
    input = torch.randint(v_size, (batch_size, max_length))
    actual = (torch.zeros(batch_size, 508).size(), torch.zeros(batch_size, class_num).size())
    embedding, output = model(input)
    predicted = (embedding.size(), output.size())
    assert actual == predicted


def test_myclassifier_2() -> None:
    batch_size = 24
    emb_dim = 300
    max_length = 256
    class_num = 5
    data_path = "/workspace/data/dataset_ja_train.json"
    ja_vector_path = "/workspace/amazon_review/weight/japanese_fasttext_vectors.vec"
    category = "home"

    text_field = data.Field(
        sequential=True,
        tokenize=tokenizer,
        use_vocab=True,
        lower=True,
        include_lengths=True,
        batch_first=True,
        fix_length=max_length,
        init_token="<cls>",
        eos_token="<eos>",
    )
    label_field = data.Field(sequential=False, use_vocab=False)
    fields = [("text", text_field), ("label", label_field)]
    df = pd.read_json(data_path, orient="record", lines=True).sample(n=10, random_state=1)
    df = df[df["product_category"] == category]
    columns = ["review_body", "stars"]
    dataset = dataframe2dataset(df, fields, columns)
    japanese_fasttext_vectors = Vectors(name=ja_vector_path)
    text_field.build_vocab(dataset, vectors=japanese_fasttext_vectors, min_freq=1)
    v_size = len(text_field.vocab.stoi)

    model = MyClassifier(emb_dim, v_size, max_length, class_num, text_field)
    input = torch.randint(v_size, (batch_size, max_length))
    actual = (torch.zeros(batch_size, 508).size(), torch.zeros(batch_size, class_num).size())
    embedding, output = model(input)
    predicted = (embedding.size(), output.size())
    assert actual == predicted


def test_mmd_multiscale_output_size():
    batch_size = 24
    input = torch.rand(batch_size, 508)
    actual = MMD("multiscale").forward(input, input).size()
    predicted = torch.rand(1).mean().size()
    assert actual == predicted


def test_mmd_rbf_output_size():
    batch_size = 24
    input = torch.rand(batch_size, 508)
    actual = MMD("rbf").forward(input, input).size()
    predicted = torch.rand(1).mean().size()
    assert actual == predicted


def test_mmd_rbf_output_value_1():
    batch_size = 24
    input = torch.rand(batch_size, 508)
    actual = MMD("rbf").forward(input, input)
    predicted = torch.tensor(0)
    assert actual == predicted


def test_rbf_output_value_2():
    input_1 = torch.tensor([[-0.1, 0.0, 0.1], [0.2, 0.3, 0.4]])
    input_2 = torch.tensor([[-0.4, -0.3, -0.2], [-0.1, 0.0, 0.1]])
    actual = MMD("rbf").forward(input_1, input_2)
    predicted = torch.tensor(0)
    assert actual == predicted
