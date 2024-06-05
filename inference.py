from model import Model
import torch
from config import config

tokenizer = config['tokenizer']
device = config['device']

def model_pred(tokens,temp):
    model,_ = Model()
    with torch.no_grad():
        pred = model(tokens,temp)
        pred = pred.view(-1, pred.shape[-1]).argmax(axis=1)
    return pred

def tokenize_text(text):
    seq_length = 300
    q_tokens = tokenizer(text,add_special_tokens=False)['input_ids']
    pad = [0 for i in range(seq_length-len(q_tokens))]
    final_tokens = [q_tokens + pad]
    last_index = len(q_tokens)-1
    
    return torch.tensor(final_tokens),last_index

def inference(text, starter='', temperature=1.0):
    curr = 0
    pred_list = []
    t, last_token = tokenize_text(text + '[CLS]' + starter)
    t = t.to(device)
    
    while curr != 102:
        print('\n',"Generating...")
        all_pred = model_pred(t, temperature)
        pred = all_pred[last_token].item()
        pred_list.append(pred)
        t[0][last_token + 1] = pred
        last_token += 1
        curr = pred
        
        if curr > 10:
            break
    print("Question from the model: ".upper(), starter + ' ' + tokenizer.decode(pred_list),'\n')

    return starter + ' ' + tokenizer.decode(pred_list)

while True:
    text = input("Enter a paragraph: ")
    inference(text, '')
