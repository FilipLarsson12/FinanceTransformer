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


# dataloader class
class DataLoader():

  # Function to normalize the prices
  def normalize(self, prices):
      min_price = np.min(prices)
      max_price = np.max(prices)
      self.min_price = min_price
      self.max_price = max_price
      normalized_prices = (prices - min_price) / (max_price - min_price)
      return normalized_prices

  def load_data_from_yfinance(self, filepath, startDate, endDate):

    # Define the stock ticker and the time period
    ticker = 'AAPL'
    start_date = startDate
    end_date = endDate

    # Download the data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Extract the closing prices
    prices = data['Close'].values

    prices = self.normalize(prices)

    # Open (or create if it doesn't exist) the file 'data.txt'
    file_path = 'data.txt'
    with open(file_path, 'w') as file:
        # Write the ticker name as a header
        file.write(f"Ticker: {ticker}\n")
        
        # Write each price point on a new line
        for price in prices:
            file.write(f"{price}\n")

    print(f"Data written to {file_path}")


  def load_data_from_file(filepath, block_size):
    # Open the file for reading
    with open(filepath, 'r') as file:
        # Read all lines
        lines = file.readlines()
        
        # Skip the first line (ticker name) and convert the remaining lines to floats
        prices = [float(line.strip()) for line in lines[1:]]

    # subtracting one because train_data and targets will both have length equal to len(prices) - 1  
    pop_elements = (len(prices)-1) % block_size
    print(f"popping {pop_elements} prices")

    if pop_elements != 0:
      prices = prices[:-pop_elements]
    prices = torch.Tensor(prices)
    return prices

  def restructure_data(self, block_size, data):
    data = data.view(-1, block_size, 1)
    return data

# ----------------------------------------------------------------------------------------

# training run


# model config class
@dataclass
class ModelConfig():
    input_dim = 1
    embd_dim = 6
    block_size = 3
    n_layers = 1

#init
model_config = ModelConfig()

model = FinanceTransformer(model_config)

# load the data into our program

dataloader = DataLoader()

data_file = "data.txt"

dataloader.load_data_from_yfinance(data_file, startDate='2010-01-01', endDate='2024-01-01')

prices = DataLoader.load_data_from_file("data.txt", model_config.block_size)

print(len(prices))

print(f"prices shape {prices.shape}")

price_inputs = prices[:-1]
targets = prices[1:]
print(len(price_inputs))

price_inputs = dataloader.restructure_data(model_config.block_size, price_inputs)
targets = dataloader.restructure_data(model_config.block_size, targets)

print(f"data going into the transformer: {price_inputs}")
print("Original input shape:", price_inputs.shape)


# define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# training loop
model.train()

epochs = 10

for epoch in range(epochs):

  # get prediction
  preds = model(price_inputs) 

  # reset gradients
  optimizer.zero_grad()

  # calculate loss
  loss = loss_fn(preds, targets)

  # calculate gradients from loss
  loss.backward()

  # update weights
  optimizer.step()

  print(f"epoch {epoch}, loss {loss}")

  if (epoch == 0):
    first_loss = loss
  if (epoch == epochs-1):
    last_loss = loss

    loss_reduction = first_loss - last_loss
    # prints
    print("Pred shape after the model processed my data: ", preds.shape)
    print("Prediction from model: ", preds)
    print(f"acual targets {targets}")
    print(f"first loss: {first_loss}, last loss: {last_loss}")
    print(f"total loss reduction in {epochs} epochs: {loss_reduction}")
