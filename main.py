import math
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Function to plot heatmap for weights
def plot_heatmap(data, title, vmin=None, vmax=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=False, cmap='coolwarm', cbar=True, center=0, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel('Weight Matrix Columns')
    plt.ylabel('Weight Matrix Rows')
    plt.show()

def plot_1d(data, title):
    plt.figure(figsize=(10, 4))
    plt.plot(data, marker='o')
    plt.title(title)
    plt.xlabel('Weight Index')
    plt.ylabel('Weight Value')
    plt.show()

# class that defines the self attention layers
class SelfAttentionLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # we use a linear layer to produce the k, q and v.
        # we use the same dimensionality for them as the embd_dim
        self.config = config
        self.embd_dim = config.embd_dim
        # must make sure that we can use several attention heads
        assert config.embd_dim % config.n_head == 0, f"Embedding dimension {config.embd_dim} is not divisible by the number of heads {config.n_head}."
        
        self.n_head = config.n_head
        self.embd_dim = config.embd_dim

        # to calculate k, q and v
        self.calc_k = nn.Linear(config.embd_dim, config.embd_dim)
        self.calc_q = nn.Linear(config.embd_dim, config.embd_dim)
        self.calc_v = nn.Linear(config.embd_dim, config.embd_dim)

        # final proj before return
        self.proj = nn.Linear(config.embd_dim, config.embd_dim)
        # register parameter for the lower triangular mask-matrix
        self.register_buffer("bias",
        torch.tril(torch.ones(config.block_size, config.block_size))
        .view(1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embd_dim

        k = self.calc_k(x)
        q = self.calc_q(x)
        v = self.calc_v(x)

        # split key, query and values into their own attention heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, number_heads, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, number_heads, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, number_heads, T, head_size)

        # calculating how much the embeddings "care" about one another
        k = k.transpose(-2, -1)
        keyquery_matrix = (q @ k) * (1.0 / math.sqrt(k.size(-1)))

        # make it impossible for embeddings to get information from embeddings that comes after
        keyquery_matrix = keyquery_matrix.masked_fill(self.bias[:, :T, :T] == 0, float('-inf'))

        keyquery_matrix = F.softmax(keyquery_matrix, dim=-1)

        # calculate updated embd_values for the embeddings based on how much information should flow between them
        x = keyquery_matrix @ v # (B, number_heads, T, T) @ (B, number_heads, T, head_size) = (B, number_heads, T, head_size)
        x = x.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

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
        self.ln_1 = nn.LayerNorm(config.embd_dim)
        self.attn = SelfAttentionLayer(config)
        self.ln_2 = nn.LayerNorm(config.embd_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = self.ln_1(x)
        x = self.attn(x)
        x = self.ln_2(x)
        x = self.mlp(x)
        return x



class Visualizer():

    def plot_preds(self, actual_prices, preds, width=10):
        actual_prices = np.array(actual_prices)
        preds = np.array(preds)

        # Create x-values for the points, ensuring a distance of 1 between each point
        x_values = np.arange(len(actual_prices))

        # Create a figure with the specified width and a default height
        plt.figure(figsize=(width, 6))

        # Plot the points with labels
        plt.plot(x_values, actual_prices, color='blue', linestyle='-', marker='o', markersize=2, linewidth=0.5, alpha=0.7, label='Actual Prices')  # '-' specifies the line style, 'o' adds points
        plt.plot(x_values, preds, color='red', linestyle='-', marker='o', markersize=2, linewidth=0.5, alpha=0.7, label='Predicted Prices')  # '-' specifies the line style, 'o' adds points

        plt.title('Stock Price Plot')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()  # Add a legend to the plot
        plt.show()


    def plot_loss(self, lossi):

      # Plot the loss
      plt.plot(lossi, label='Training Loss')
      plt.xlabel('Iteration')
      plt.ylabel('Loss')
      plt.title('Training Loss over Iterations')
      plt.legend()
      plt.show()



# dataloader class
class DataLoader():

  def __init__(self, B, T):
    self.B = B
    self.T = T
    # this holds the complete training data
    self.data = []
    self.currentPosition = 0
    # this is annoying but we need it to prevent input -> target mapping from
    # last price of one ticker to first price of next ticker
    self.currentTicker = 0
    self.nTickers = 0
    # state to keep track of epochs in training
    self.dataPerEpoch = 0

  # Function to normalize the prices using z-normalization
  def normalize(self, prices):
      mean_price = np.mean(prices)
      std_price = np.std(prices)
      normalized_prices = (prices - mean_price) / std_price
      return normalized_prices

  def load_data_from_yfinance(self, tickerList, filepath, startDate, endDate):
      with open(filepath, 'w') as file:
          for ticker in tickerList:
              # download data from Yahoo Finance
              data = yf.download(ticker, start=startDate, end=endDate)
              prices = data['Close'].values

              # normalize prices of the ticker
              normalized_prices = self.normalize(prices)

              # popping prices if they dont modulo with my model block_size
              # subtracting one because inputs and targets will have len - 1 elements
              pop_elements = (len(normalized_prices)-1) % (self.T*self.B)
              print(f"\npopping {pop_elements} prices from ticker {ticker} when loading data into dataloader")
              if pop_elements != 0:
                normalized_prices = normalized_prices[:-pop_elements]

              # write prices to file, file's purpose is for me to look at and sanity check
              file.write(f"--- Ticker: {ticker} ---\n")
              for price in normalized_prices:
                  file.write(f"{price}\n")

              # add ticker prices to self.data, increment self.nTickers and update self.dataPerEpoch
              self.data.append(normalized_prices.tolist())
              self.nTickers += 1
              self.dataPerEpoch += len(normalized_prices) - 1

  def next_batch(self):

    # pluck out next batch
    inp = self.data[self.currentTicker][self.currentPosition : self.currentPosition + self.B * self.T]
    targets = self.data[self.currentTicker][self.currentPosition + 1 : self.currentPosition + (self.B * self.T) + 1]

    # convert to tensors and reshape, also add extra dim so shape becomes [B, T, 1]
    inp = torch.Tensor(inp).view(self.B, self.T).unsqueeze(-1)
    targets = torch.Tensor(targets).view(self.B, self.T).unsqueeze(-1)

    # update pointer
    self.currentPosition += self.B * self.T



    # if next batch is out of bounds reset pointer
    if self.currentPosition + self.B * self.T + 1 > len(self.data[self.currentTicker]):
      self.currentPosition = 0
      # if we are at the last ticker go to first ticker again
      if self.currentTicker == (self.nTickers - 1):
        self.currentTicker = 0
      # else go to next ticker
      else:
        self.currentTicker += 1


    return inp, targets



# model class
class FinanceTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.block_size is not None
        self.block_size = config.block_size
        self.price_embeddings = nn.Linear(config.input_dim, config.embd_dim)
        self.position_embeddings = nn.Embedding(config.block_size, config.embd_dim)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_normlayer = nn.LayerNorm(config.embd_dim)
        self.predictionlayer = nn.Linear(config.embd_dim, 1)

        # Apply custom weight initialization
        # self.apply(self.custom_weight_init)

    # Method to calculate the number of parameters
    def calculate_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params


    # will see if I change this, currently this init actually scales up weights which can be seen from the heatmap plots
    def custom_weight_init(self, module):
        if isinstance(module, nn.Linear):
            fan_in = module.weight.data.size(1)  # number of input units
            std = 1.0 / math.sqrt(fan_in)
            plot_heatmap(module.weight.data.cpu().numpy(), f'Original weights for {module}', vmin=-1.0, vmax=1.0)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            plot_heatmap(module.weight.data.cpu().numpy(), f'Scaled weights for {module}', vmin=-1.0, vmax=1.0)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            fan_in = module.weight.data.size(1) # number of inputs units
            std = 1.0 / math.sqrt(fan_in)
            plot_heatmap(module.weight.data.cpu().numpy(), f'Original weights for {module}')
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            plot_heatmap(module.weight.data.cpu().numpy(), f'Scaled weights for {module}')

    def plot_heatmap_all_or_specific(self, plot_type='weights', module_name='all', vmin=None, vmax=None):
        """
        Plot heatmap for weights or gradients for all modules or a specific module.

        Args:
            plot_type (str): 'weights' to plot weights, 'grads' to plot gradients.
            module_name (str): 'all' to plot all modules or specify the module name.
            vmin (float): Minimum value for color bar.
            vmax (float): Maximum value for color bar.
        """
        for name, module in self.named_modules():
            if module_name != 'all' and name != module_name:
                continue

            data = None
            if plot_type == 'weights':
                if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding) or isinstance(module, nn.LayerNorm):
                    data = module.weight.detach().cpu().numpy()
            elif plot_type == 'grads':
                if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding) or isinstance(module, nn.LayerNorm):
                    if module.weight.grad is not None:
                        data = module.weight.grad.cpu().detach().numpy()

            if data is not None:
                if data.ndim == 2:
                    plot_heatmap(data, f'{plot_type.capitalize()} for {name}', vmin, vmax)
                elif data.ndim == 1:
                    plot_1d(data, f'{plot_type.capitalize()} for {name}')
                else:
                    print(f"No {plot_type} data for {name} or data is not 1D/2D.")
            else:
                print(f"No {plot_type} data for {name}.")

            if module_name != 'all':
                break
        else:
            if module_name != 'all':
                print(f"Module {module_name} not found in the model.")


    def forward(self, x, targets=None):
        _, T, _ = x.size()

        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

        # input dim is: (batch_am, block_size, 1)

        pos = torch.arange(0, T, dtype=torch.long) # shape (T)
        pos_emb = self.position_embeddings(pos) # position embeddings of shape (T, embd_dim)
        price_emb = self.price_embeddings(x) # price embeddings of shape (B, T, embd_dim)

        # adding positional embeddings to the price embeddings
        x = price_emb + pos_emb # pos updated price embeddings of shape (B, T, embd_dim)

        for i, block in enumerate(self.layers):
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


# model config class
@dataclass
class ModelConfig():
    input_dim = 1
    embd_dim = 6
    block_size = 6
    n_layers = 1
    n_head = 2

# ----------------------------------------------------------------------------------------

#init
model_config = ModelConfig()

model = FinanceTransformer(model_config)

# get the size of the model
num_parameters = model.calculate_parameters()
print(f"Number of parameters in the model: {num_parameters}")

# define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

vliser = Visualizer()

# track losses
losses = []


# load the data into our program
dataLoader = DataLoader(B=2, T=model_config.block_size)

data_file = "data.txt"

tickerList = ['MSFT']

dataLoader.load_data_from_yfinance(tickerList, data_file, startDate='2015-01-01', endDate='2015-07-01')

print(f"dataloader.dataPerEpoch: {dataLoader.dataPerEpoch}")

model.train()

epochs = 1000

batch_size = dataLoader.B * dataLoader.T

# calculate amount of batches in one epoch
total_batches = dataLoader.dataPerEpoch*epochs // batch_size

print(f"total batches: {total_batches}")

# turn on/off prints
prints = False

# training run

print("----- START OF TRAINING -----")

with tqdm(total=(total_batches), desc=f"Training progress") as pbar:

  for i in range(total_batches):

      # load batch
      X, Y = dataLoader.next_batch()

      # get prediction
      preds, loss = model(X, Y)

      if loss is not None:
        # reset gradients
        optimizer.zero_grad()

        # calculate gradients from loss
        loss.backward()

        # update weights
        optimizer.step()

        # append the loss 
        losses.append(loss.item())

        # Update the progress bar
        pbar.update(1)

      if prints:
        print(f"Price Embedding Layer Gradients: {model.price_embeddings.weight.grad}")
        print(f"Position Embedding Layer Gradients: {model.position_embeddings.weight.grad}")


print("----- END OF TRAINING -----")


# let's visualize how our model performs
model.eval()

print(f"losses: {losses}")
vliser.plot_loss(losses)


# plot the preds for one ticker at a time
for i in range(len(tickerList)):

  # pluck out all data for current ticker
  tickerData = dataLoader.data[i][:-1]

  print(f"tickerdata before {tickerData}")

  # reshape to (B, T, 1) for model inference
  tickerData = torch.Tensor(tickerData).view(-1, model_config.block_size, 1)

  print(f"tickerdata after torchification {tickerData}")

  with torch.no_grad():
    tickerPreds, _ = model(tickerData)


  # pluck out real targets
  tickerTargets = dataLoader.data[i][1:]

  # reshape to 1D for plot
  tickerPreds = tickerPreds.view(-1)

  vliser.plot_preds(actual_prices=tickerTargets, preds=tickerPreds, width=16)
