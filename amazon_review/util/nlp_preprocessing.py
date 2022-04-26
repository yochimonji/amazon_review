import re
import string

from janome.tokenizer import Tokenizer
from torchtext.legacy import data

janome_tokenizer = Tokenizer()


def tokenizer_ja(text):
    return [token for token in janome_tokenizer.tokenize(text, wakati=True)]


def tokenizer_en(s):
    # 記号はスペースで置換して除去
    for p in string.punctuation:
        s = s.replace(p, " ")

    # 連続する空白は1つにする
    s = re.sub(r" +", r" ", s).strip()

    # スペースで分割
    return s.split()


def dataframe2dataset(df, fields, columns):
    examples = []
    for _, row in df.iterrows():
        df_list = [row[column] for column in columns]
        examples += [data.Example.fromlist(df_list, fields)]
    dataset = data.Dataset(examples, fields)
    return dataset
