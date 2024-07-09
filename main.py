import math
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import yfinance as yf
import pandas as pd
import numpy as np

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
        .view(1, 1, config.block_size, config.block_size))


    def forward(self, x):

        B, T, C = x.size() # batch size, sequence length, embd_dim           

        kqv = self.calc_kqv(x)

        # splitting so I have key, query and value
        k, q, v = kqv.split(self.config.embd_dim, dim=-1)

        # calculating how much the embeddings "care" about one another
        # i.e calculating how much information should flow between the different embeddings
        k = k.transpose(-2, -1)
        keyquery_matrix = (q @ k) * (1.0 / math.sqrt(k.size(-1)))
        print(f"keyquery matrix shape {keyquery_matrix.shape}")
        # make it impossible for embeddings to get information from embeddings that comes after
        print(f"bias: {self.bias[:, :, :T, :T]}")
        keyquery_matrix = keyquery_matrix.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        print(f"keyquery matrix after applying bias {keyquery_matrix}")

        keyquery_matrix = F.softmax(keyquery_matrix, dim=-1)
        print(f"keyquery matrix after applying bias + softmax {keyquery_matrix}")

        # calculate updated embd_values for the embeddings based on how much information should flow between them
        x = keyquery_matrix @ v # (batch_am, seq_len, seq_len) @ (batch_am, seq_len, embd_dim) = (batch_am, seq_len, embd_dim)

        # final proj
        x = self.proj(x)

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

    def forward(self, x, targets=None):

        
        _, T, _ = x.size()

        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # input dim is: (batch_am, block_size, 1)
        x = self.embedding(x) # (batch_am, block_size, embd_dim)
        for block in self.layers:
            x = block(x) # (batch_am, block_size, embd_dim)
        x = self.final_normlayer(x) # (batch_am, block_size, embd_dim)

        if targets is not None:
          # now we get predictions at every position in the sequence in every batch
          preds = self.predictionlayer(x) # (batch_am, block_size, 1)
          loss_fn = nn.MSELoss()
          loss = loss_fn(preds, targets)
        else:
          preds = self.predictionlayer(x) # (batch_am, block_size, 1)
          loss = None
        
        return preds, loss

    # to generate from the model
    def generate(self, context, max_new_prices):
      


      pred = self(context)
      if pred.dim() == 4:
        pred = pred[:, -1, -1, -1]
      elif pred.dim() == 3:
        pred = pred[-1, -1, -1]

      return pred

      

# dataloader class
class DataLoader():

  def __init__(self, config, batch_size):
    self.tickerList = []
    self.config = config
    self.batch_size = batch_size
    
    # used to keep track of what batch to return when load_next_batch() is called
    self.currentTickerIndex = 0
    self.indexWithinTicker = 0
    self.all_price_inputs = None
    self.all_targets = None
    self.batches_per_ticker = 0
    # keep track of which epoch we are in
    self.currentEpoch = 1
    self.currentBatch = 0

  # Function to normalize the prices
  def normalize(self, prices):
      min_price = np.min(prices)
      max_price = np.max(prices)
      self.min_price = min_price
      self.max_price = max_price
      normalized_prices = (prices - min_price) / (max_price - min_price)
      return normalized_prices

  def load_data_from_yfinance(self, tickerList, filepath, startDate, endDate):
      with open(filepath, 'w') as file:
          for ticker in tickerList:
              data = yf.download(ticker, start=startDate, end=endDate)
              prices = data['Close'].values
              normalized_prices = self.normalize(prices)

              # popping prices if they dont modulo with my model block_size
              pop_elements = (len(normalized_prices)-1) % self.config.block_size
              print(f"popping {pop_elements} prices")
              if pop_elements != 0:
                normalized_prices = normalized_prices[:-pop_elements]

              file.write(f"Ticker: {ticker}\n")
              for price in normalized_prices:
                  file.write(f"{price}\n")
              file.write("\n")
              self.tickerList.append(ticker)


  def load_data_from_file(self, filepath, block_size):
    data_dict = {}
    current_ticker = None

    # Open the file for reading
    with open(filepath, 'r') as file:
        # Read all lines
        lines = file.readlines()

        for line in lines:
          line = line.strip()
          if not line:
            continue
          if line.startswith("Ticker:"):
            current_ticker = line.split("Ticker: ")[1]
            data_dict[current_ticker] = []
          else:
            data_dict[current_ticker].append(float(line))

    return data_dict


  def load_next_batch(self):
     
    # go to next ticker if next batch exceeds remaining current ticker data
    if self.indexWithinTicker+self.batch_size > self.batches_per_ticker:
      # if we're at the last ticker go to first ticker and increment epoch
      if self.currentTickerIndex == (len(self.tickerList) - 1):
        self.currentTickerIndex = 0
        self.currentEpoch += 1

      else: 
        self.currentTickerIndex += 1
      self.indexWithinTicker = 0
    
    # placeholders
    outerIdx = self.currentTickerIndex
    innerIdx = self.indexWithinTicker

    # pluck out next batch
    inputs = self.all_price_inputs[outerIdx, innerIdx:innerIdx+self.batch_size]
    targets = self.all_targets[outerIdx, innerIdx:innerIdx+self.batch_size]

    # update index within current ticker
    self.indexWithinTicker = self.indexWithinTicker + self.batch_size

    # update batch tracker
    self.currentBatch += 1

    return inputs, targets

  # function to sanity check calculations in case I forget the code in the future
  def confirm_inputs_targets_match(self, inputs, targets):
    # in case we have multiple tickers
    if inputs.dim() == 4:
      inputs_reshaped = inputs.view(inputs.shape[0], -1)
      targets_reshaped = targets.view(targets.shape[0], -1)
      inputs_reshaped = inputs_reshaped[:, 1:]
      targets_reshaped = targets_reshaped[:, :-1]
      
    # in case we only have one ticker
    else:
      inputs_reshaped = inputs_reshaped.view(-1)
      targets_reshaped = targets_reshaped.view(-1)
      inputs_reshaped = inputs_reshaped[:, 1:]
      targets_reshaped = targets_reshaped[:, :-1]


    return torch.equal(inputs_reshaped, targets_reshaped)



  # helper function to check if next batch is in this or next epoch
  def check_for_next_epoch(self):

      if (self.indexWithinTicker+self.batch_size > self.batches_per_ticker) and (self.currentTickerIndex == (len(self.tickerList) - 1)):
        return True

      return False


  def restructure_data(self, data_dict):
        block_size = self.config.block_size
        price_inputs_list = []
        targets_list = []

        for ticker, prices in data_dict.items():

            # create inputs and targets that have equal indices in their corresponding matrix
            ticker_data_price_inputs = prices[:-1]
            ticker_data_targets = prices[1:]

            # convert them to torch.Tensors
            ticker_data_price_inputs = torch.Tensor(ticker_data_price_inputs)
            ticker_data_targets = torch.Tensor(ticker_data_targets)

            print(f"ticker_data_price_inputs before reshape: {ticker_data_price_inputs.shape}")
            print(f"ticker_data_targets before reshape: {ticker_data_targets.shape}")

            # reshape so that each row is of length block_size
            ticker_data_price_inputs = ticker_data_price_inputs.view(-1, block_size, 1)
            ticker_data_targets = ticker_data_targets.view(-1, block_size, 1)

            print(f"ticker_data_price_inputs after reshape: {ticker_data_price_inputs.shape}")
            print(f"ticker_data_targets after reshape: {ticker_data_targets.shape}")

            # append the price points and their targets of the current ticker to the overarching data structure
            price_inputs_list.append(ticker_data_price_inputs)
            targets_list.append(ticker_data_targets)

        # stack the datqa from every ticker on top of each other so they become a batch dimension
        self.all_price_inputs = torch.stack(price_inputs_list)
        self.all_targets = torch.stack(targets_list)

        # now the shape of the data will be: (number_of_tickers, block_size, 1)
        print(f"combined_price_inputs final shape: {self.all_price_inputs.shape}")
        print(f"combined_targets final shape: {self.all_targets.shape}")

        self.batches_per_ticker = self.all_price_inputs.shape[1]

        return self.all_price_inputs, self.all_targets

# ----------------------------------------------------------------------------------------

# training run


# model config class
@dataclass
class ModelConfig():
    input_dim = 1
    embd_dim = 5
    block_size = 3
    n_layers = 1

#init
model_config = ModelConfig()

model = FinanceTransformer(model_config)

# load the data into our program

dataLoader = DataLoader(model_config, batch_size=4)

data_file = "data.txt"

dataLoader.load_data_from_yfinance(['TLSA'], data_file, startDate='2015-01-01', endDate='2024-01-01')

data = dataLoader.load_data_from_file("data.txt", model_config.block_size)

price_inputs, targets = dataLoader.restructure_data(data)

# define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# training loop
model.train()

epochs = 1

while True:

  # placeholder
  epoch = dataLoader.currentEpoch

  # load batch
  X, Y = dataLoader.load_next_batch()
  
  # get prediction
  preds, loss = model(X, Y)

  if loss is not None:
    # reset gradients
    optimizer.zero_grad()

    # calculate gradients from loss
    loss.backward()

    # update weights
    optimizer.step()
  else:

    # get last prediction from model
    pred = preds[:, -1, -1, -1]
    print(f"Prediction from model: {pred}")



  print(f"epoch {epoch}, loss {loss}")

  if (dataLoader.currentBatch == 1):
    # this is first batch
    first_loss = loss

  if (epoch == epochs and dataLoader.check_for_next_epoch()):
    # this is the last batch
    last_loss = loss
    loss_reduction = first_loss - last_loss

    # prints
    print("Pred shape after the model processed my data: ", preds.shape)
    print(f"first loss: {first_loss}, last loss: {last_loss}")
    print(f"total loss reduction in {epochs} epochs: {loss_reduction}")
    # stop training
    break

# ensure test has dimensions: ticker_am, batch_am, <= block_size, 1
test = torch.Tensor(
    [
        [[0.5], [0.6], [0.7]]
     ]
    )
print(f"test dim : \n{test.shape}")
pred = model(test)
print(f"input: {test}, pred \n{pred}")

