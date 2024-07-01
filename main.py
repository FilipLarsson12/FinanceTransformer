import torch
import torch.nn as nn

# class that defines the self attention layers
class SelfAttentionLayer(nn.Module):

    def __init__(self, config):
        # we use a linear layer to produce the k, q and v.
        # we use the same dimensionality for them as the embd_dim
        self.calc_kqv = nn.Linear(config.embd_dim, config.embd_dim * 3)


# model class
class FinanceTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.input_dim, config.embd_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        return x

# create some data and reshape it so that it can be processed by the model
data = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]])
print("Original input shape:", data.shape)
data = data.view(1, 10, 1)
print("After reshaping to (batch_size, sequence_length, input_dim):", data.shape)

class Config():
    def __init__(self, input_dim, embd_dim):
        self.input_dim = input_dim
        self.embd_dim = embd_dim
    
model_config = Config(1, 32)

model = FinanceTransformer(model_config)

data = model(data)

print("Data shape after being processed by the model: ", data.shape)
print(data)

