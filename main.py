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

        # to calculate k, q and v, had trouble with k and q getting zero grads with elegant method so thinking of switching to regular

        if config.elegant_kqv:
            self.calc_kqv = nn.Linear(config.embd_dim, 3 * config.embd_dim)
        else:
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
        print(f"Before going into SelfAttentionLayer: {x}")

        B, T, C = x.size() # batch size, sequence length, embd_dim
        print(f"B: {B}, T: {T}, C: {C}")

        if self.config.elegant_kqv:
            kqv = self.calc_kqv(x)
            k, q, v = kqv.split(self.embd_dim, dim=2)
        else:
            k = self.calc_k(x)
            q = self.calc_q(x)
            v = self.calc_v(x)

        print(f"Keys (k): {k}")
        print(f"Queries (q): {q}")
        print(f"Values (v): {v}")

        # calculating how much the embeddings "care" about one another
        k = k.transpose(-2, -1)
        keyquery_matrix = (q @ k) * (1.0 / math.sqrt(k.size(-1)))
        print(f"keyquery matrix shape: {keyquery_matrix.shape}")
        print(f"keyquery matrix before applying bias: {keyquery_matrix}")

        # make it impossible for embeddings to get information from embeddings that comes after
        print(f"bias: {self.bias[:, :T, :T]}")
        keyquery_matrix = keyquery_matrix.masked_fill(self.bias[:, :T, :T] == 0, float('-inf'))
        print(f"keyquery matrix after applying bias: {keyquery_matrix}")

        keyquery_matrix = F.softmax(keyquery_matrix, dim=-1)
        print(f"keyquery matrix after applying bias + softmax: {keyquery_matrix}")

        # calculate updated embd_values for the embeddings based on how much information should flow between them
        x = keyquery_matrix @ v # (batch_am, seq_len, seq_len) @ (batch_am, seq_len, embd_dim) = (batch_am, seq_len, embd_dim)
        print(f"Updated x after attention: {x}")

        # final proj
        x = self.proj(x)
        print(f"Output of SelfAttentionLayer after proj: {x}")

        return x


# class that defines a MLP
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.proj_1 = nn.Linear(config.embd_dim, 4 * config.embd_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.proj_2 = nn.Linear(config.embd_dim * 4, config.embd_dim)

    def forward(self, x):
        print(f"Before going into MLP: {x}")
        x = self.proj_1(x)
        print(f"After proj_1: {x}")
        x = self.gelu(x)
        print(f"After GELU: {x}")
        x = self.proj_2(x)
        print(f"After proj_2 (Output of MLP): {x}")
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
        print(f"Before going into Block LayerNorm 1: {x}")
        x = self.ln_1(x)
        print(f"After Block LayerNorm 1: {x}")
        x = self.attn(x)
        print(f"After Block Attention Layer: {x}")
        x = self.ln_2(x)
        print(f"After Block LayerNorm 2: {x}")
        x = self.mlp(x)
        print(f"After Block MLP (Output of Block): {x}")
        return x



class Visualizer():

    def plot_preds(self, actual_prices, preds, width=10, downsample_factor=1):
        actual_prices = np.array(actual_prices)
        preds = np.array(preds)

        # Downsample the data if downsample_factor > 1
        if downsample_factor > 1:
            actual_prices = actual_prices[::downsample_factor]
            preds = preds[::downsample_factor]

        # Create a figure with the specified width and a default height
        plt.figure(figsize=(width, 6))

        # Plot the points with labels
        plt.plot(actual_prices, color='blue', linestyle='-', marker='o', markersize=2, linewidth=0.5, alpha=0.7, label='Actual Prices')  # '-' specifies the line style, 'o' adds points
        plt.plot(preds, color='red', linestyle='-', marker='o', markersize=2, linewidth=0.5, alpha=0.7, label='Predicted Prices')  # '-' specifies the line style, 'o' adds points

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

  def __init__(self, config, batch_size):
    self.tickerList = []
    self.config = config
    self.batch_size = batch_size
    self.data_dict = {}

    # used to keep track of what batch to return when load_next_batch() is called
    self.currentTickerIndex = 0
    self.indexWithinTicker = 0
    self.all_price_inputs = None
    self.all_targets = None
    self.batches_per_ticker = 0
    # keep track of which epoch we are in
    self.currentEpoch = 1
    self.currentBatch = 0

  # Function to normalize the prices using z-normalization
  def normalize(self, prices):
      mean_price = np.mean(prices)
      std_price = np.std(prices)
      self.mean_price = mean_price
      self.std_price = std_price
      normalized_prices = (prices - mean_price) / std_price
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
            self.data_dict[current_ticker] = []
          else:
           self.data_dict[current_ticker].append(float(line))


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

  # function to load data for plotting model's predictions, this data will overlap which the load_next_batch() can't do
  def load_data_for_plot_inference(self, ticker, amount):
    # this function will return data prepared for model in the shape: (batch_am, block_size, 1)
    # and data in the shape (amount) that will be plotted against model output to evaluate performance
    block_size = self.config.block_size

    assert amount % block_size == 0, f"number of data points must be divisible by block_size of {block_size}"

    data = self.data_dict[ticker][:amount]
    original_tensor = torch.tensor(data)

    num_contexts = len(original_tensor) - block_size + 1

    # Initialize the result tensor
    result = []

    # Create the windows
    for i in range(num_contexts):
        context = original_tensor[i:i+block_size].unsqueeze(1)
        result.append(context)

    # Stack the windows to create the final tensor
    result_tensor = torch.stack(result)
    print(f"result tensor shape {result_tensor.shape}")
    print(f"result tensor \n{result_tensor} \n original tensor \n{original_tensor}")

    return result_tensor, original_tensor



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


  def restructure_data(self):
        block_size = self.config.block_size
        price_inputs_list = []
        targets_list = []

        for ticker, prices in self.data_dict.items():

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
        print(f"Before going into Price Embedding Layer: {x}")

        _, T, _ = x.size()
        print(f"Sequence length T: {T}")

        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

        # input dim is: (batch_am, block_size, 1)


        pos = torch.arange(0, T, dtype=torch.long) # shape (T)
        pos_emb = self.position_embeddings(pos) # position embeddings of shape (T, embd_dim)
        print(f"positional embeddings: {pos_emb}")
        price_emb = self.price_embeddings(x) # price embeddings of shape (B, T, embd_dim)
        print(f"price embeddings: {price_emb}")

        # adding positional embeddings to the price embeddings
        x = price_emb + pos_emb # pos updated price embeddings of shape (B, T, embd_dim)

        print(f"After Price + Pos Embedding Layer: {x}")

        for i, block in enumerate(self.layers):
            print(f"Before going into Block {i}: {x}")
            x = block(x) # (batch_am, block_size, embd_dim)
            print(f"After Block {i}: {x}")

        x = self.final_normlayer(x) # (batch_am, block_size, embd_dim)
        print(f"After Final LayerNorm: {x}")

        if targets is not None:
            # now we get predictions at every position in the sequence in every batch
            preds = self.predictionlayer(x) # (batch_am, block_size, 1)
            print(f"Predictions: {preds}")
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds, targets)
            print(f"Loss: {loss}")
        else:
            preds = self.predictionlayer(x) # (batch_am, block_size, 1)
            print(f"Predictions: {preds}")
            loss = None

        return preds, loss


    # to generate next price point from the model or to generate multiple price points for plotting for example
    def generate(self, context, multiple=None):
        print(f"Input to generate: {context}")

        if multiple:
            print("Generate multiple")
            preds, _ = self(context)
            print(f"Preds shape: {preds.shape}")
            preds = preds[:, -1, -1]
            print(f"Output preds: {preds}")
            return preds

        else:
            pred, _ = self(context)
            print(f"Single pred before reshape: {pred}")
            pred = pred[-1, -1, -1]
            print(f"Output pred: {pred}")
            return pred

# ----------------------------------------------------------------------------------------


# model config class
@dataclass
class ModelConfig():
    input_dim = 1
    embd_dim = 4
    block_size = 3
    n_layers = 1
    elegant_kqv = False

#init
model_config = ModelConfig()

model = FinanceTransformer(model_config)



# training loop
model.train()

# define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

vliser = Visualizer()

# track losses
lossi = []


'''


# alt training run for experiments

X = torch.Tensor([
    [[0.4], [2.2]]
])

Y = torch.Tensor([
     [[2.2], [1.6]]
])

for i in range(20):

  # get prediction
  preds, loss = model(X, Y)

  if loss is not None:
    # reset gradients
    optimizer.zero_grad()

    # calculate gradients from loss
    loss.backward()


    if (i % 10 == 0):
      if model_config.elegant_kqv:
        model.plot_heatmap_all_or_specific('grads', 'layers.0.attn.calc_kqv')
      else:
        model.plot_heatmap_all_or_specific(plot_type='grads')


    # update weights
    optimizer.step()
    lossi.append(loss.item())

with torch.no_grad():

  Preds, _ = model(X)


Y = Y.view(-1)
Preds = Preds.view(-1)
print(f"Y shape {Y.shape}")
print(f"Preds shape {Preds.shape}")

Preds = torch.squeeze(Preds, dim=0)

print(f"Prices: {Y}")
print(f"Preds: {Preds}")

# Plot the loss
plt.plot(lossi, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss over Iterations')
plt.legend()
plt.show()

vliser.plot_preds(Y, Preds, width=16)

'''


# training run

# load the data into our program

dataLoader = DataLoader(model_config, batch_size=4)

data_file = "data.txt"

dataLoader.load_data_from_yfinance(['MSFT'], data_file, startDate='2015-01-01', endDate='2015-04-01')

dataLoader.load_data_from_file("data.txt", model_config.block_size)

price_inputs, targets = dataLoader.restructure_data()

print("START OF TRAINING -------------")



epochs = 32

print(f"total: {(dataLoader.batches_per_ticker // dataLoader.batch_size)*len(dataLoader.tickerList)*epochs}")

with tqdm(total=(dataLoader.batches_per_ticker // dataLoader.batch_size)*len(dataLoader.tickerList)*epochs, desc=f"Training progress") as pbar:

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

      lossi.append(loss.item())

      # Update the progress bar
      pbar.update(1)

      if (loss.item() < 0.01):
        break

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
      print(f"Price Embedding Layer Gradients: {model.price_embeddings.weight.grad}")
      print(f"Position Embedding Layer Gradients: {model.position_embeddings.weight.grad}")

      # stop training
      break

  print("END OF TRAINING -------------")


# let's visualize how our model performs

with torch.no_grad():

  price_inputs = price_inputs.squeeze(0)


  print(f"price_inputs shape before doing inference: {price_inputs.shape}")

  preds, _ = model(price_inputs)

price_inputs = price_inputs.view(price_inputs.shape[0]*price_inputs.shape[1])
preds = preds.view(preds.shape[0]*preds.shape[1])

print(f"price_inputs shape {price_inputs.shape}")

print(f"preds shape {preds.shape}")

print(f"price_inputs {price_inputs}")
print(f"preds {preds}")

print(f"losses: {lossi}")

vliser.plot_loss(lossi)

vliser.plot_preds(price_inputs, preds, width=16)
