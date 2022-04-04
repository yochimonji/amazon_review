import torch
from torch import nn


class MyClassifier(nn.Module):
    def __init__(self, emb_dim, v_size, max_length, class_num, text_field):
        super().__init__()
        self.embed = nn.Embedding(v_size, emb_dim)
        self.embed.weight.data.copy_(text_field.vocab.vectors)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * max_length, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(64, class_num),
        )

    def forward(self, sentence):
        embed_sentence = self.embed(sentence)
        embed_sentence = embed_sentence.view(embed_sentence.shape[0], -1)
        output = self.mlp(embed_sentence)
        return output


class MyEmbed(nn.Module):
    def __init__(self, emb_dim, v_size, text_field):
        super().__init__()
        self.embed = nn.Embedding(v_size, emb_dim)
        self.embed.weight.data.copy_(text_field.vocab.vectors)

    def forward(self, sentence):
        embed_sentence = self.embed(sentence)
        embed_sentence = embed_sentence.view(embed_sentence.shape[0], -1)
        return embed_sentence


class MyMlp(nn.Module):
    def __init__(self, emb_dim, max_length, class_num):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * max_length, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(64, class_num),
        )

    def forward(self, embed_sentence):
        output = self.classifier(embed_sentence)
        return output


class MLPWithMMD(nn.Module):
    def __init__(self, emb_dim, v_size, max_length, class_num, text_field):
        super().__init__()
        self.embed = nn.Embedding(v_size, emb_dim)
        self.embed.weight.data.copy_(text_field.vocab.vectors)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * max_length, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(64, class_num),
        )

    def forward(self, sentence):
        embed_sentence = self.embed(sentence)
        embed_sentence = embed_sentence.view(embed_sentence.shape[0], -1)
        output = self.mlp(embed_sentence)
        return embed_sentence, output
