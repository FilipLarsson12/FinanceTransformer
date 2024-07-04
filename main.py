import math
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F

# class that defines the self attention layers
class SelfAttentionLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # we use a linear layer to produce the k, q and v.
        # we use the same dimensionality for them as the embd_dim
        self.config = config
        # to calculate k, q and v
        self.calc_kqv = nn.Linear(config.embd_dim, config.embd_dim * 3)
        # final proj before return
        self.proj = nn.Linear(config.embd_dim, config.embd_dim)
        # register parameter for the lower triangular mask-matrix
        self.register_buffer("bias", 
        torch.tril(torch.ones(config.block_size, config.block_size))
        .view(1, config.block_size, config.block_size))
        

    def forward(self, x):
        print("x that produces kqv: ", x)
        kqv = self.calc_kqv(x)
        print("kqv shape: ", kqv.shape)
        print("kqv: ", kqv)
        # splitting so I have key, query and value 
        k, q, v = kqv.split(self.config.embd_dim, dim=-1)
        print("k shape: ", k.shape)
        print("q shape: ", q.shape)
        print("v shape: ", v.shape)
        # calculating how much the embeddings "care" about one another
        # i.e calculating how much information should flow between the different embeddings
        k = k.transpose(-2, -1)
        keyquery_matrix = (q @ k) * (1.0 / math.sqrt(k.size(-1)))
        print("Keyquery_matrix before mask: \n", keyquery_matrix)
        # make it impossible for embeddings to get information from embeddings that comes after
        keyquery_matrix = keyquery_matrix.masked_fill(self.bias == 0, float('-inf'))

        print("Key: \n", k)
        print("Query: \n", q)
        print("Keyquery_matrix after mask: \n", keyquery_matrix)
        print("Keyquery_matrix shape: \n", keyquery_matrix.shape)
        keyquery_matrix = F.softmax(keyquery_matrix, dim=-1)
        print("Keyquery_matrix after mask and softmax: \n", keyquery_matrix)
        print("Value: \n", v)
        print("Value shape: ", v.shape)

        # calculate updated embd_values for the embeddings based on how much information should flow between them
        x = keyquery_matrix @ v # (batch_am, seq_len, seq_len) @ (batch_am, seq_len, embd_dim) = (batch_am, seq_len, embd_dim)

        # final proj
        x = self.proj(x)

        print("x: \n", x)

        return x
    

# class that defines a MLP
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.proj_1 = nn.Linear(config.embd_dim, 4 * config.embd_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.proj_2 = nn.Linear(config.embd_dim * 4, config.embd_dim)
    
    def forward(self, x):
        x = self.proj_1(x)
        x = self.gelu(x)
        x = self.proj_2(x)
        return x
    

# class that defines a "block" in the model, it contains a layernorm to normalize activations, attention layer, MLP layer and another layernorm

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.normlayer_1 = nn.LayerNorm(config.embd_dim)
        self.attentionlayer = SelfAttentionLayer(config)
        self.normlayer_2 = nn.LayerNorm(config.embd_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = self.normlayer_1(x)
        print("x after normlayer inside block: ", x)
        x = self.attentionlayer(x)
        x = self.normlayer_2(x)
        x = self.mlp(x)
        return x


# model class
class FinanceTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.input_dim, config.embd_dim)
        # self.position_embeddings = nn.Embedding(config.block_size, config.embd_dim)

        self.layers = nn.ModuleList(
            [
                Block(config) for _ in range(config.n_layers)
            ]    
        )
        self.final_normlayer = nn.LayerNorm(config.embd_dim)
        self.predictionlayer = nn.Linear(config.embd_dim, 1)

    def forward(self, x):
        # input dim is: (batch_am, seq_len, 1)
        x = self.embedding(x) # (batch_am, seq_len, embd_dim)
        print("Embeddings from FT: ", x)
        for block in self.layers:
            x = block(x) # (batch_am, seq_len, embd_dim)
        print("x after FT blocks: ", x)
        x = self.final_normlayer(x) # (batch_am, seq_len, embd_dim)
        print("x before final proj: ", x)
        x = self.predictionlayer(x) # (batch_am, seq_len, 1)
        # now we have predictions at every position in the sequence in every batch
        return x


# ----------------------------------------------------------------------------------------

# training run

# create some data and reshape it so that it can be processed by the model
data = torch.tensor([
    [0.1, 0.7, 2.3],
    [0.4, 0.5, 1.8],
    [0.3, 1.2, 1.5],
    [4.1, 0.3, 2.7],
    [3.2, 1.8, 4.5],
    [2.9, 2.1, 3.3],
    [0.2, 3.0, 4.0],
    [1.0, 2.8, 0.9],
    [4.3, 4.5, 1.1],
    [1.5, 2.2, 0.4]
])




# model config class
@dataclass
class ModelConfig():
    input_dim = 1
    embd_dim = 6
    block_size = 3
    n_layers = 1

#init
model_config = ModelConfig()

data = data.view(data.shape[0], data.shape[1], 1)

print(f"data going into the transformer: {data}")
print("Original input shape:", data.shape)

model = FinanceTransformer(model_config)

pred = model(data) 

# prints

print("Pred shape after the model processed my data: ", pred.shape)
print("Prediction from model: ", pred)


test = torch.tensor([[ 0.1732,  0.3001, 0.5],
         [ 0.1069,  0.4810, 0.1],
         [-0.0701,  0.9635, 1]])
testlayer = nn.LayerNorm(3)
res = testlayer(test)
print("res: ", res)