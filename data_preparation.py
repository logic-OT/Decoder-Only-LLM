import pandas as pd
from config import config
from torch import IntTensor, LongTensor,utils
import random
from tqdm import tqdm

tokenizer = config["tokenizer"]

def tokenize_input(qa):
    #1. tokenizing with a max seq length of 300 and padding layers
    #2. Adding an <sos> and <eos> token to target values. In this case; [CLS] and [SEP]
    seq_length = 300
    q_tokens = tokenizer(qa[0]['value'],add_special_tokens=False)['input_ids']
    a_tokens = tokenizer(qa[1]['value'],padding=True)['input_ids']


    x_tokens = q_tokens + a_tokens[:-1]
    y_tokens = q_tokens[1:] + a_tokens

    x_pad = [0 for i in range(seq_length-len(x_tokens))]
    y_pad = [0 for i in range(seq_length-len(x_tokens))]
    final_x = x_tokens + x_pad
    final_y = y_tokens + y_pad

    return final_x, final_y

def prepare_data():
    print("Preparing data")
    data_path = config['data_path']

    data = pd.read_json(data_path).to_dict(orient='list')

    # Tokenizing all data
    tokens = []
    targets = []
    # Wrap the iterable with tqdm to show a progress bar
    for i in tqdm(random.sample(data['conversation'], len(data['conversation'])), desc="Tokenizing data"):
        try:
            x, y = tokenize_input(i)

            if len(x) == 300:
                tokens.append(x)
                targets.append(y)
        except:
            pass

    X = IntTensor(tokens)
    Y = LongTensor(targets)
    dataset = utils.data.TensorDataset(X, Y)

    return dataset


