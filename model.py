#necessary imports
import numpy as np
import torch
import matplotlib.pyplot as plot
import torch.nn.functional as F
from transformers import AutoModel
from config import config

device = config['device']
tokenizer = config['tokenizer']

class embed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = AutoModel.from_pretrained('bert-base-uncased')

    def forward(self,x_tokens):
        inputs = {'input_ids':x_tokens}
        with torch.no_grad():
            attention_mask = (inputs['input_ids'] != 0).int()
            outputs = self.embedder(**inputs,attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
        return embeddings


class pos_enc(torch.nn.Module):
    ## sinusoidal positional encoding
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x):
        batch_size, max_seq_length, dmodel = x.shape
        pe = torch.zeros_like(x) #position encoding matrix

        # Compute the positional encoding values
        for pos in range(max_seq_length):
            for i in range(0, dmodel):
                if i % 2 == 0:
                    pe[:, pos, i] = torch.math.sin(pos / (10000 ** (2 * i / dmodel)))
                else:
                    pe[:, pos, i] = torch.math.cos(pos / (10000 ** (2 * i / dmodel)))

        x = x + pe
        return x

class self_attention(torch.nn.Module):
        def __init__(self,no_of_heads: int ,shape: tuple, mask: bool=False, QKV: list=[]):
                '''
        Initializes a Self Attention module as described in the "Attention is all you need" paper.
        This module splits the input into multiple heads to allow the model to jointly attend to information
        from different representation subspaces at different positions. After attention is applied independently
        on each head, the module concatenates and linearly transforms the results.




        ## Parameters:
            * no_of_heads (int): Number of attention heads. To implement single head attention, set this parameter to 1. ,i.e, no_of_heads = 1

           * shape (tuple): A tuple (seq_length, dmodel) where `seq_length` is the length of the input sequence,
                           and `dmodel` is the dimensionality of the input feature space.

            * mask (bool, optional): If True, a mask will be applied to prevent attention to future positions. This is particularly useful in decoder layers to ensure that the predictions for a sequence position can depend only on the known outputs at previous positions. Defaults to False.

            * QKV (list, optional): A list containing pre-computed Query (Q), Key (K), and Value (V) matrices. If provided, these matrices will be used instead of computing `Q`, `K`, and `V` from the input tensor. This is useful for operations where `Q`, `K`, and `V` come from different sources, such as in cross-attention in the Transformer decoder. The list should contain three tensors of shape (batch_size, seq_length, dmodel), corresponding to Q, K, and V, respectively.
        The forward pass computes the multi-head attention for input `x` and returns the transformed output.
                '''
                super().__init__()
                self.h = no_of_heads
                self.seq_length,self.dmodel = shape
                self.dk = self.dmodel//self.h
                self.softmax = torch.nn.Softmax(dim=-1)
                self.mQW = torch.nn.ModuleList([torch.nn.Linear(self.dmodel,self.dk) for i in range(self.h)])
                self.mKW = torch.nn.ModuleList([torch.nn.Linear(self.dmodel,self.dk) for i in range(self.h)])
                self.mVW = torch.nn.ModuleList([torch.nn.Linear(self.dmodel,self.dk) for i in range(self.h)])
                self.output_linear = torch.nn.Linear(self.dmodel,self.dmodel)
                self.mask = mask
                self.QKV = QKV

        def __add_mask(self,atten_values):
              #masking attention values
              mask_value = -torch.inf
              mask = torch.triu(torch.ones(atten_values.shape) * mask_value, diagonal=1)
              masked = atten_values + mask.to(device)
              return masked

        def forward(self, x):
            heads = []
            for i in range(self.h):
                # Apply linear projections in batch from dmodel => h x d_k
                if self.QKV:
                      q = self.mQW[i](self.QKV[0])
                      k = self.mKW[i](self.QKV[1])
                      v = self.mVW[i](self.QKV[2])
                else:
                        q = self.mQW[i](x)
                        k = self.mKW[i](x)
                        v = self.mVW[i](x)


                # Calculate attention using the projected vectors q, k, and v
                self.scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.dk, dtype=torch.float32))
                if self.mask:
                      self.scores = self.__add_mask(self.scores)


                attn = self.softmax(self.scores)
                head_i = torch.matmul(attn, v)

                heads.append(head_i)

            # Concatenate all the heads together
            multi_head = torch.cat(heads, dim=-1)
            # Final linear layer
            output = self.output_linear(multi_head)

            return output + x  # Residual connection
        

class decoder_layer(torch.nn.Module):
    def __init__(self,shape: tuple,no_of_heads:int = 1):
        '''
        Implementation of Transformer Dencoder
        Parameters:
            shape (tuple): The shape (H, W) of the input tensor
            no_of_heads (int): number of heads in the attention mechanism. set this to 1 for single head attntion. default = 1
        Returns:
            Tensor: The output of the encoder layer after applying attention, feedforward network, and normalization.
        '''
        super().__init__()

        self.max_seq_length,self.dmodel = shape
        def ff_weights():
            layer1 =  torch.nn.Linear(self.dmodel,600)
            layer2 = torch.nn.Linear(600,600)
            layer3 = torch.nn.Linear(600,self.dmodel)
            return layer1,layer2,layer3

        self.no_of_heads = no_of_heads

        self.multi_head =  self_attention(no_of_heads=no_of_heads, mask=True,
                                                    shape=(self.max_seq_length,self.dmodel))

        self.layer1,self.layer2,self.layer3 = ff_weights()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.layerNorm = torch.nn.LayerNorm(shape)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def feed_forward(self,x):
        f = self.layer1(x)
        f = self.relu1(f)
        f = self.layer2(f)
        f = self.relu2(f)
        f = self.layer3(f)

        return self.layerNorm(f  + x) #residual connection

    def forward(self,x):
        x = self.multi_head(x)
        x = self.layerNorm(x)
        x = self.feed_forward(x)
        x = self.layerNorm(x)

        return x

class architecture(torch.nn.Module):
    def __init__(self,n_classes,shape) -> None:
        super().__init__()
        self.max_seq_length,self.dmodel = shape
        self.projected_dmodel = 224
        self.embedding_layer = embed()
        self.proj_to_224 = torch.nn.Linear(self.dmodel, self.projected_dmodel)
        self.positional = pos_enc()
        self.decoder1 = decoder_layer(shape=(self.max_seq_length,self.projected_dmodel), no_of_heads=8)
        self.decoder2 = decoder_layer(shape=(self.max_seq_length,self.projected_dmodel), no_of_heads=8)
        self.decoder3 = decoder_layer(shape=(self.max_seq_length,self.projected_dmodel), no_of_heads=8)
        self.decoder4 = decoder_layer(shape=(self.max_seq_length,self.projected_dmodel), no_of_heads=8)
        self.decoder5 = decoder_layer(shape=(self.max_seq_length,self.projected_dmodel), no_of_heads=8)
        self.decoder6 = decoder_layer(shape=(self.max_seq_length,self.projected_dmodel), no_of_heads=8)
        self.decoder7 = decoder_layer(shape=(self.max_seq_length,self.projected_dmodel), no_of_heads=8)
        self.decoder8 = decoder_layer(shape=(self.max_seq_length,self.projected_dmodel), no_of_heads=8)
        # self.decoder5 = decoder_layer(shape=(self.max_seq_length,self.projected_dmodel))
        self.final_MLP = torch.nn.Linear(self.projected_dmodel,n_classes)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self,x,temperature=1.0):
        x = self.embedding_layer(x)
        x = self.proj_to_224(x)
        x = self.positional(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        x = self.decoder5(x)
        x = self.decoder6(x)
        x = self.decoder7(x)
        x = self.decoder8(x)
        x = self.final_MLP(x)
        logits = x / temperature
        x = self.softmax(logits)


        return x
    
def Model():
    vocab_size = tokenizer.vocab_size
    model = architecture(n_classes = vocab_size, shape = (300,768))
    model = model.to(device)
    if config["model_weights"] is not None:
        try:
            model.load_state_dict(torch.load(config["model_weights"]))
        except:
             model.load_state_dict(torch.load(config["model_weights"],map_location=torch.device('cpu')))
    return model,vocab_size

