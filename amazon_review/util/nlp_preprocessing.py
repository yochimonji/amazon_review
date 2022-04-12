from janome.tokenizer import Tokenizer
from torchtext.legacy import data

janome_tokenizer = Tokenizer()


def tokenizer(text):
    return [token for token in janome_tokenizer.tokenize(text, wakati=True)]


def dataframe2dataset(df, fields, columns):
    examples = []
    for _, row in df.iterrows():
        df_list = [row[column] for column in columns]
        examples += [data.Example.fromlist(df_list, fields)]
    dataset = data.Dataset(examples, fields)
    return dataset
