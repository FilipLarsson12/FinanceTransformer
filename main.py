import math
import torch
import torch.nn as nn
from dataclasses import dataclass
# class that defines the self attention layers
class SelfAttentionLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # we use a linear layer to produce the k, q and v.
        # we use the same dimensionality for them as the embd_dim
        self.config = config
        self.calc_kqv = nn.Linear(config.embd_dim, config.embd_dim * 3)

        # register parameter for the lower triangular mask-matrix
        self.register_buffer("bias", 
        torch.tril(torch.ones(config.block_size, config.block_size))
        .view(1, config.block_size, config.block_size))


    def forward(self, x):
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
        print("Key: ", k)
        print("Query: ", q)
        print("Keyquery_matrix: ", keyquery_matrix)
        print("Keyquery_matrix shape: ", keyquery_matrix.shape)


        return x

# model class
class FinanceTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.input_dim, config.embd_dim)
        self.attentionlayer = SelfAttentionLayer(config)

    def forward(self, x):
        x = self.embedding(x)
        x = (self.attentionlayer(x))
        return x

# create some data and reshape it so that it can be processed by the model
data = torch.tensor([[0.1, 0.3, 0.5]])
print("Original input shape:", data.shape)
data = data.view(1, 3, 1)
print("After reshaping to (batch_size, sequence_length, input_dim):", data.shape)

# model config class
@dataclass
class Config():
    input_dim = 1
    embd_dim = 2
    block_size = 3
#init
model_config = Config()

model = FinanceTransformer(model_config)

data = model(data)

print("Data shape after being processed by the model: ", data.shape)


