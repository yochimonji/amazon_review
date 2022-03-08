import pandas as pd
import torch
import torchtext
from janome.tokenizer import Tokenizer
from torch import nn
from torchtext.legacy import data
from torchtext.vocab import FastText
from tqdm import tqdm

from util import init_device

device = init_device()
