from transformers import AutoTokenizer
import torch

config = {
    "data_path":"sample_data.json",
    "tokenizer": AutoTokenizer.from_pretrained('bert-base-uncased'),
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "model_weights": None,
    "learning_rate":0.0001,
    "num_batches":10,
}

