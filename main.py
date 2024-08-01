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
from typing import Literal

# Function to plot heatmap for weights
def plot_heatmap(ax, data, module, module_name, activity):
    # Ensure data is 2D for plotting
    if data.ndim == 1:
        data = data.reshape(-1, 1)  # Convert to a single row
    elif data.ndim == 3:
        data = data.reshape(-1, data.shape[-1])
        # special handling for output and grad output matrix for better visualization
    if (activity == "output" or activity == "grad output") and ((not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Embedding) and not isinstance(module, nn.GELU)) and module_name != "predictionlayer"):
        data = data.T
    # if 2D data is small enough print the weight values in the plot as well
    if data.shape[0] < 10 and data.shape[1] < 10:
      sns.heatmap(data, annot=True, fmt=".3f", cmap='coolwarm', cbar=True, center=0, ax=ax)
    else:
      sns.heatmap(data, annot=False, cmap='coolwarm', cbar=True, center=0, ax=ax)

    ax.set_title(f"{activity} for {module_name}")
    ax.set_xlabel('Weight Matrix Columns')
    ax.set_ylabel('Weight Matrix Rows')



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
        # flag for weight scaling to counteract residual streams increasing variance in the forward pass
        self.proj.RESIDUAL_SCALING_INIT = 1

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
        self.act_fn = nn.GELU(approximate='tanh') # Loss = 1.01 after 500 epochs
        #self.act_fn = nn.LeakyReLU() # Loss = 0.16 after 500 epochs
        self.proj_2 = nn.Linear(config.embd_dim * 4, config.embd_dim)
        # flag for weight scaling to counteract residual streams increasing variance in the forward pass
        self.proj_2.RESIDUAL_SCALING_INIT = 1

    def forward(self, x):
        x = self.proj_1(x)
        x = self.act_fn(x)
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
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Visualizer():


    def plot_preds(self, actual_prices, preds, title, width=10):
        actual_prices = np.array(actual_prices)
        preds = np.array(preds)

        # Create x-values for the points, ensuring a distance of 1 between each point
        x_values = np.arange(len(actual_prices))

        # Create a figure with the specified width and a default height
        plt.figure(figsize=(width, 6))

        # Plot the points with labels
        plt.plot(x_values, actual_prices, color='blue', linestyle='-', marker='o', markersize=2, linewidth=0.5, alpha=0.7, label='Actual Prices')  # '-' specifies the line style, 'o' adds points
        plt.plot(x_values, preds, color='red', linestyle='-', marker='o', markersize=2, linewidth=0.5, alpha=0.7, label='Predicted Prices')  # '-' specifies the line style, 'o' adds points

        plt.title(title)
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()  # Add a legend to the plot
        plt.show()


    def plot_graph(self, list, title, xlabel, ylabel, scale=None, width=10, height=6, padding_factor=0.0, y_ticks=None):

      plt.figure(figsize=(width, height))

      # Calculate min and max with padding
      min_val = min(list)
      max_val = max(list)
      min_limit = min_val - padding_factor * min_val
      max_limit = max_val + padding_factor * max_val

      if scale == 'log':
        plt.yscale('log')

      elif scale == 'symlog':
        plt.yscale('symlog', linthresh=1e-15)

      plt.ylim(min_limit, max_limit)

      # Set y-axis ticks for more granularity if provided
      if y_ticks is not None:
          plt.yticks(y_ticks)

      # Plot the loss
      plt.plot(list, label=title)
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
      plt.legend()
      plt.show()


    # function that accesses the module_activities dict of a modelObserver and plots the submodule activities
    def plot_module_activities(self, modelObserver):
        for module_name, module_activity in modelObserver.module_activities.items():
            print(f"Plotting module activity for {module_name}")

            module = module_activity.pop('module', None)


            forward_keys = ['input', 'weights', 'bias', 'output']
            backward_keys = ['grad input', 'grad output', 'weight grads', 'bias grads']

            forward_activities = {k: v for k, v in module_activity.items() if k in forward_keys and v is not None}
            backward_activities = {k: v for k, v in module_activity.items() if k in backward_keys and v is not None}

            n_forward_activities = len(forward_activities)
            n_backward_activities = len(backward_activities)

            #print(f"Number of forward activities: {n_forward_activities}")
            #print(f"Number of backward activities: {n_backward_activities}")

            plot_width = max(n_forward_activities, n_backward_activities)

            fig, axes = plt.subplots(2, plot_width, figsize=(5 * plot_width, 10))

            if n_forward_activities > 0:
                # Plot forward activities
                for i, (key, data) in enumerate(forward_activities.items()):
                    # print(f"Plotting forward activity {i+1}/{n_forward_activities}: {key}")
                    if data is not None:
                        plot_heatmap(axes[0, i], data, module, module_name, key)  # Pass module to plot_heatmap
                    else:
                        print("Data is None")
                for i in range(n_forward_activities, plot_width):
                    # Hide the subplot
                    axes[0, i].set_visible(False)

            if n_backward_activities > 0:
                # Plot backward activities
                for i, (key, data) in enumerate(backward_activities.items()):
                    # print(f"Plotting backward activity {i+1}/{n_backward_activities}: {key}")
                    if data is not None:
                        plot_heatmap(axes[1, i], data, module, module_name, key)  # Pass module to plot_heatmap
                    else:
                        print("Data is None")
                for i in range(n_backward_activities, plot_width):
                    # Hide the subplot
                    axes[1, i].set_visible(False)

            plt.tight_layout()
            plt.show()



# normalizer class capable of using 2 different normalization schemes. Z-normalization as well as percentage-normalization
class DataNormalizer():

  def __init__(self, norm_scheme):
      self.norm_scheme = norm_scheme
      # dictionary to save mean and std for each ticker for z-denormalization
      self.tickers_mean_std = {}
      # dictionary to save initial prices for each ticker for percentage-denormalization
      self.initial_prices = {}
      # state to keep track of training period so we use consistent z-normalization of train and val set
      self.trainStartDate = None
      self.trainEndDate = None

      self.tickers_min_max = {}

  def normalize(self, tickername, prices, split):
    if self.norm_scheme == "z":
      return self.z_normalize(tickername, prices, split)
    elif self.norm_scheme == "percentage":
      return self.percentage_normalize(tickername, prices, split)
    elif self.norm_scheme == "min_max":
      return self.min_max_normalize(tickername, prices, split)


  def de_normalize(self, tickername, normalized_prices):
    if self.norm_scheme == "z":
      return self.z_denormalize(tickername, normalized_prices)
    elif self.norm_scheme == "percentage":
      return self.percentage_denormalize(tickername, normalized_prices)
    elif self.norm_scheme == "min_max":
      return self.min_max_denormalize(tickername, normalized_prices)


  # Function to normalize the prices using z-normalization
  def z_normalize(self, tickername, prices, split="train"):
      # use the mean and std of train set to normalize train and val set
      if split == "train":
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        # save the mean and std for the ticker
        self.tickers_mean_std[tickername] = (mean_price, std_price)
      elif split == "val":
        # we use the training period to normalize the val set's prices (for example if we trained on 2010-2015 data 
        # we use this period's mean and std when normalizing the val ticker's prices)
        if self.trainStartDate is None or self.trainEndDate is None:
          raise ValueError("Training period not set. Cannot normalize val data.")
        if tickername not in self.tickers_mean_std:
          data = yf.download(tickername, self.trainStartDate, self.trainEndDate)
          prices_for_mean_and_std = data['Close'].values
          mean_price = np.mean(prices_for_mean_and_std)
          std_price = np.std(prices_for_mean_and_std)
          # save the mean and std for the ticker
          self.tickers_mean_std[tickername] = (mean_price, std_price)
        else:
          mean_price, std_price = self.tickers_mean_std[tickername]

      normalized_prices = (prices - mean_price) / std_price
      return normalized_prices


  # Function to reverse the z-normalization
  def z_denormalize(self, tickername, normalized_prices):
      if tickername not in self.tickers_mean_std:
          raise ValueError(f"Statistics for ticker {tickername} not found.")
      mean_price, std_price = self.tickers_mean_std[tickername]
      original_prices = (normalized_prices * std_price) + mean_price
      return original_prices
    

  # Function to normalize the prices using percentage change
  def percentage_normalize(self, tickername, prices, split="train"):

      self.initial_prices[tickername] = prices[0]

      # Normalize each price by the first price
      normalized_prices = prices / prices[0]
      normalized_prices = np.log(normalized_prices)

      return normalized_prices

  def percentage_denormalize(self, tickername, normalized_prices):
      if tickername not in self.initial_prices:
          raise ValueError(f"Initial price for ticker {tickername} not found.")
      normalized_prices = np.exp(normalized_prices)

      # Denormalize prices
      original_prices = normalized_prices * self.initial_prices[tickername]
      
      return original_prices


  # Function to normalize the prices using min-max scaling
  def min_max_normalize(self, tickername, prices, split="train"):
      if split == "train":
          min_price = np.min(prices)
          max_price = np.max(prices)
          # save the min and max for the ticker
          self.tickers_min_max[tickername] = (min_price, max_price)
      elif split == "val":
          min_price = np.min(prices)
          max_price = np.max(prices)
          # min_price, max_price = self.tickers_min_max[tickername]
          self.tickers_min_max[tickername] = (min_price, max_price)

      normalized_prices = 2 * (prices - min_price) / (max_price - min_price) - 1
      return normalized_prices


  # Function to reverse the min-max normalization
  def min_max_denormalize(self, tickername, normalized_prices):
      if tickername not in self.tickers_min_max:
          raise ValueError(f"Min and max for ticker {tickername} not found.")
      min_price, max_price = self.tickers_min_max[tickername]
      original_prices = (normalized_prices + 1) / 2 * (max_price - min_price) + min_price
      return original_prices


# dataloader class
class DataLoader():

  def __init__(self, model_config, normalizer):
    self.B = model_config.batch_size
    self.T = model_config.block_size
    # this holds the complete training data
    self.trainData = []
    # this holds the complete val data
    self.valData = []
    self.currentPosition = 0
    # this is annoying but we need it to prevent input -> target mapping from
    # last price of one ticker to first price of next ticker
    self.currentTicker = 0
    self.nTrainTickers = 0
    self.nValTickers = 0
    # state to keep track of epochs in training
    self.dataPerEpoch = 0
    # initialize normalizer
    self.normalizer = normalizer


  def load_data_from_yfinance(self, tickerList, filepath, startDate, endDate, split="train"):
      # keep track of training period to use for z-normalizing test prices later
      if split == "train":
          self.normalizer.trainStartDate = startDate
          self.normalizer.trainEndDate = endDate

      with open(filepath, 'w') as file:
          for ticker in tickerList:
              # download data from Yahoo Finance
              data = yf.download(ticker, start=startDate, end=endDate)
              if data.empty:
                raise ValueError(f"No data found for ticker {ticker} between {startDate} and {startDate}.")
              prices = data['Close'].values

              # normalize prices
              normalized_prices = self.normalizer.normalize(ticker, prices, split)


              # popping prices if they dont modulo with my model block_size
              # subtracting one because inputs and targets will have len - 1 elements
              if split == "train":
                pop_elements = (len(normalized_prices)-1) % (self.T*self.B)
              # we don't care about batch size if we load in test set data, we only care about block size
              elif split == "val":
                pop_elements = (len(normalized_prices)-1) % self.T

              print(f"\npopping {pop_elements} prices from ticker {ticker} when loading data into dataloader")
              if pop_elements != 0:
                normalized_prices = normalized_prices[:-pop_elements]

              # write prices to file, file's purpose is for me to look at and sanity check
              file.write(f"--- Ticker: {ticker} ---\n")
              for price in normalized_prices:
                  file.write(f"{price}\n")

              # add ticker prices to self.data, increment self.nTickers and update self.dataPerEpoch
              if split == "train":
                self.trainData.append(normalized_prices.tolist())
                self.nTrainTickers += 1
                self.dataPerEpoch += len(normalized_prices) - 1

              elif split == "val":
                self.nValTickers += 1
                self.valData.append(normalized_prices.tolist())


  def next_batch(self):

    # pluck out next batch
    inp = self.trainData[self.currentTicker][self.currentPosition : self.currentPosition + self.B * self.T]
    targets = self.trainData[self.currentTicker][self.currentPosition + 1 : self.currentPosition + (self.B * self.T) + 1]

    # convert to tensors and reshape, also add extra dim so shape becomes [B, T, 1]
    inp = torch.Tensor(inp).view(self.B, self.T).unsqueeze(-1)
    targets = torch.Tensor(targets).view(self.B, self.T).unsqueeze(-1)

    # update pointer
    self.currentPosition += self.B * self.T

    # if next batch is out of bounds reset pointer
    if self.currentPosition + self.B * self.T + 1 > len(self.trainData[self.currentTicker]):
      self.currentPosition = 0
      # if we are at the last ticker go to first ticker again
      if self.currentTicker == (self.nTrainTickers - 1):
        self.currentTicker = 0
      # else go to next ticker
      else:
        self.currentTicker += 1

    return inp, targets



# class that can track and store model activities
class ModelObserver():

    def __init__(self, model):
        self.module_activities = {}
        self.hook_handles = []
        self.model = model

    # add forward and backward hooks to all submodules of the model
    def register_hooks(self, save_grads=False, only_save_modules=None):
        for name, module in self.model.named_modules():
            # if only_save_modules is None we save activities for all modules else we only save for modules in only_save_modules list
            if only_save_modules is None or name in only_save_modules:
                if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding, nn.GELU, nn.LeakyReLU)):
                    forward_handle = module.register_forward_hook(lambda m, i, o, n=name: self.save_module_activity_forward(n, m, i, o))
                    self.hook_handles.append(forward_handle)
                    if save_grads:
                        if module is not self.model.position_embeddings and module is not self.model.price_embeddings:
                            backward_handle = module.register_full_backward_hook(lambda m, gi, go, n=name: self.save_module_activity_backward(n, m, gi, go))
                            self.hook_handles.append(backward_handle)

                        # special handling for price and pos embeddings
                        else:
                            backward_handle = module.weight.register_hook(lambda grad, m=module, n=name: self.save_grad(m, n, grad, 'weight grads'))
                            self.hook_handles.append(backward_handle)
                            if hasattr(module, 'bias') and module.bias is not None:
                                emb_bias_handle = module.bias.register_hook(lambda grad, m=module, n=name: self.save_grad(m, n, grad, 'bias grads'))
                                self.hook_handles.append(emb_bias_handle)


    # special function to save grads for weight and bias for price and pos embeddings
    def save_grad(self, module, name, grad, grad_type):
        if name not in self.module_activities:
            self.module_activities[name] = {}
        self.module_activities[name]['module'] = module  # Save the module object
        self.module_activities[name][grad_type] = grad.detach()


    # remove all the forward and backward hooks
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


    # capture all module activity in the forward pass
    def save_module_activity_forward(self, module_name, module, input, output):

        weights = module.weight.detach() if hasattr(module, 'weight') else None
        bias = module.bias.detach() if hasattr(module, 'bias') and module.bias is not None else None
         # input is a tuple so have to extract tensor
        input = input[0]
        input = input.detach()
        output = output.detach()
        self.module_activities[module_name] = {}
        self.module_activities[module_name]['module'] = module  # Save the module object
        self.module_activities[module_name]['input'] = input
        # only add weights and bias if they are not None
        if weights is not None:
          self.module_activities[module_name]['weights'] = weights
        if bias is not None:
          self.module_activities[module_name]['bias'] = bias
        self.module_activities[module_name]['output'] = output


    # capture all module activity in the backward pass
    def save_module_activity_backward(self, module_name, module, grad_input, grad_output):
        # check that activities exists
        weight_grads = module.weight.grad.detach() if hasattr(module, 'weight') and module.weight.grad is not None else None
        bias_grads = module.bias.grad.detach() if hasattr(module, 'bias') and module.bias.grad is not None else None
        # grad_input and grad_output is a tuple so have to extract tensor
        grad_input = grad_input[0]
        grad_output = grad_output[0]
        grad_input = grad_input.detach() if grad_input is not None else None
        grad_output = grad_output.detach() if grad_output is not None else None

        # add to activity dict
        if grad_input is not None:
          self.module_activities[module_name]['grad input'] = grad_input
        if weight_grads is not None:
          self.module_activities[module_name]['weight grads'] = weight_grads
        if bias_grads is not None:
          self.module_activities[module_name]['bias grads'] = bias_grads
        if grad_input is not None:
          self.module_activities[module_name]['grad output'] = grad_output


    # method that returns a dictionary with module and avg values for the activities given to the function
    def compute_module_statistics(self, activityList, pooling="avg"):
      network_statistics = {}
      # loop through the submodules
      for module_name, activities in self.module_activities.items():
        # init empty dict for current module
        network_statistics[module_name] = {}
        # fill it with avg values for activity in activityList
        for activity, value in activities.items():
          if activity in activityList:
            if pooling == "avg":
              network_statistics[module_name][f'{activity} avg'] = value.mean()
            elif pooling == "var":
              if value.numel() > 1:  # Check if the tensor has more than 1 element
                network_statistics[module_name][f'{activity} var'] = torch.var(value, unbiased=False)

      return network_statistics



# model class
class FinanceTransformer(nn.Module):

    def __init__(self, config, plot_activity=False):
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
        self.apply(self.custom_weight_init)


    # Method to calculate the number of parameters
    def calculate_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params


    # will see if I change this, currently this init actually scales up weights which can be seen from the heatmap plots
    def custom_weight_init(self, module):
        if isinstance(module, nn.Linear):
            fan_in = module.weight.data.size(1)  # number of input units
            std = 1.0 / math.sqrt(fan_in)
            if hasattr(module, 'RESIDUAL_SCALING_INIT'):
              std *= (2 * self.config.n_layers) ** -0.5
            # plot_heatmap(module.weight.data.cpu().numpy(), f'Original weights for {module}')
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # plot_heatmap(module.weight.data.cpu().numpy(), f'Scaled weights for {module}')

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            fan_in = module.weight.data.size(1) # number of inputs units
            std = 1.0 / math.sqrt(fan_in)
            # plot_heatmap(module.weight.data.cpu().numpy(), f'Original weights for {module}')
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # plot_heatmap(module.weight.data.cpu().numpy(), f'Scaled weights for {module}')


    def forward(self, idx, targets=None):
        _, T, _ = idx.size()

        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

        # input dim is: (batch_am, block_size, 1)

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.position_embeddings(pos) # position embeddings of shape (T, embd_dim)
        price_emb = self.price_embeddings(idx) # price embeddings of shape (B, T, embd_dim)

        # adding positional embeddings to the price embeddings
        x = price_emb + pos_emb # pos updated price embeddings of shape (B, T, embd_dim)

        for i, block in enumerate(self.layers):
            x = block(x) # (batch_am, block_size, embd_dim)

        x = self.final_normlayer(x) # (batch_am, block_size, embd_dim)

        if targets is not None:
            # now we get predictions at every position in the sequence in every batch
            preds = self.predictionlayer(x) # (batch_am, block_size, 1)
            if self.config.loss_fn == "L1":
              loss_fn = nn.L1Loss()
            elif self.config.loss_fn == "MSE":
              loss_fn = nn.MSELoss()
            loss = loss_fn(preds, targets)
        else:
            preds = self.predictionlayer(x) # (batch_am, block_size, 1)
            loss = None

        return preds, loss


# model config class
@dataclass
class ModelConfig():
    input_dim: int = 1
    batch_size: int = 4
    block_size: int = 16
    embd_dim: int = 16
    n_layers: int = 8
    n_head: int = 4
    loss_fn: Literal['L1', 'MSE'] = "MSE"

# ----------------------------------------------------------------------------------------

# autodetect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#init
model_config = ModelConfig()

model = FinanceTransformer(model_config)
model.to(device)

# object to track model activities
modelObserver = ModelObserver(model)

# get the size of the model
num_parameters = model.calculate_parameters()
print(f"Number of parameters in the model: {num_parameters}")

# define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

vliser = Visualizer()

# track losses and grad norms
train_losses = []
val_losses = []
# used so I can compare losses between different normalization schemes
normalized_val_losses = []
# used to inspect training stability
grad_norms = []

# load the data into our program
normalizer = DataNormalizer(norm_scheme="min_max")
dataLoader = DataLoader(model_config, normalizer)

# Training specification
trainTickerList = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'JPM', 'V', 'JNJ', 'QCOM', 'NVDA']
train_data_file = "train_data.txt"
train_startDate = '2018-08-01'
train_endDate = '2023-08-01'

# Load the train set into dataloader
dataLoader.load_data_from_yfinance(trainTickerList, train_data_file, startDate=train_startDate, endDate=train_endDate, split="train")

# Validation specification
valTickerList = ['BABA', 'NFLX', 'PFE', 'AMD', 'NVDA', 'AAPL']
val_data_file = "val_data.txt"
val_startDate = '2023-08-02'
val_endDate = '2024-07-01'

# Load the validation set into dataloader
dataLoader.load_data_from_yfinance(valTickerList, val_data_file, startDate=val_startDate, endDate=val_endDate, split="val")


# how many batches between each val loss measurement
val_interval = 50
# record lowest val loss
lowest_val_loss = float('inf')
lowest_normalized_val_loss = float('inf')
total_absolute_price_diff = 0

print(f"dataloader.dataPerEpoch: {dataLoader.dataPerEpoch}")

epochs = 20

batch_size = dataLoader.B * dataLoader.T

# calculate amount of batches in one epoch
total_batches = dataLoader.dataPerEpoch*epochs // batch_size

print(f"total batches: {total_batches}")

# toggle prints and plots
prints = False
plots = False

# register hooks so modelObserver can save model state during forward and backward pass
modelObserver.register_hooks(save_grads=True)

print(f"Model config: {model_config}")

# training run

print("\n----- START OF TRAINING -----")

with tqdm(total=(total_batches), desc=f"Training progress") as pbar:

  for i in range(total_batches):

      # set the model to train mode
      model.train()

      # load batch
      X, Y = dataLoader.next_batch()

      # move batch to correct device
      X = X.to(device)
      Y = Y.to(device)

      # get prediction
      preds, train_loss = model(X, Y)

      if train_loss is not None:
        # reset gradients
        optimizer.zero_grad()

        # calculate gradients from loss
        train_loss.backward()

        # clipping the gradients if they get too high values
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update weights
        optimizer.step()

        # append the training loss
        train_losses.append(train_loss.item())
        grad_norms.append(norm.item())

        # Update the progress bar
        pbar.update(1)

      # eval during training
      if i % val_interval == 0:
        model.eval()

        val_loss = 0
        # enables comparison between losses from different data normalization schemes
        normalized_val_loss = 0

        for tickerIndex in range(dataLoader.nValTickers):

          # pluck out all data for ticker
          tickerData = dataLoader.valData[tickerIndex][:-1]
          tickerTargets = dataLoader.valData[tickerIndex][1:]

          # reshape to (-1, T, 1) for model inference
          tickerData = torch.Tensor(tickerData).view(-1, model_config.block_size, 1).to(device)
          tickerTargets = torch.Tensor(tickerTargets).view(-1, model_config.block_size, 1).to(device)
          
          # keep track of the variance of the data
          data_variance = torch.var(tickerTargets, unbiased=False).item()

          with torch.no_grad():
            tickerPreds, val_loss_ticker = model(tickerData, tickerTargets)

          # add to the overall loss
          val_loss += val_loss_ticker
          normalized_val_loss += (val_loss_ticker / data_variance)


          # plot tickers if we are recording the last validation loss
          if (i+val_interval) >= total_batches:
            # reshape to 1D for plot
            tickerPreds = tickerPreds.view(-1).cpu().numpy()
            tickerTargets = tickerTargets.view(-1).cpu().numpy()

            deNormalize_prices = True
            if deNormalize_prices:
              # call the dataloaders denormalize() function to reverse normalization for model preds and targets of current ticker
              current_ticker = valTickerList[tickerIndex]

              tickerPreds = normalizer.de_normalize(current_ticker, tickerPreds)
              tickerTargets = normalizer.de_normalize(current_ticker, tickerTargets)

              element_wise_diff = tickerPreds - tickerTargets
              absolute_diff = np.abs(element_wise_diff)
              sum_absolute_diff = np.sum(absolute_diff)
              total_absolute_price_diff += sum_absolute_diff
              
            vliser.plot_preds(actual_prices=tickerTargets, preds=tickerPreds, title=f"Val Ticker: {valTickerList[tickerIndex]}, period: {val_startDate} to {val_endDate}, processed {int((i / total_batches)*100)}% of training data", width=16)


        # calculate average loss over tickers
        val_loss = val_loss / dataLoader.nValTickers
        normalized_val_loss = normalized_val_loss / dataLoader.nValTickers
        # append the test loss
        val_losses.append(val_loss.item())
        normalized_val_losses.append(normalized_val_loss.item())

        # check if this is lowest loss
        if val_loss.item() < lowest_val_loss:
          lowest_val_loss = val_loss
        if normalized_val_loss.item() < lowest_normalized_val_loss:
          lowest_normalized_val_loss = normalized_val_loss

      # plot the preds for one ticker at a time
      if i == (total_batches-1):

        # vliser.plot_module_activities(modelObserver)
        # now we want to plot the variance of grads and output throughout the network
        all_weight_grads_var = []
        all_bias_grads_var = []
        all_output_var = []
        pooling = "var"
        network_averages = modelObserver.compute_module_statistics(['weights', 'bias', 'output'], pooling=pooling)
        print(f"network_averages: {network_averages}")
        for module, averages in network_averages.items():
          for averagename, averagevalue in averages.items():
            if averagename == f"weights {pooling}":
              all_weight_grads_var.append(averagevalue.cpu().item())
            elif averagename == f"bias {pooling}":
              all_bias_grads_var.append(averagevalue.cpu().item())
            elif averagename == f"output {pooling}":
              all_output_var.append(averagevalue.cpu().item())
        vliser.plot_graph(all_weight_grads_var, "weight var from start to end of network", "module number", "weights var", padding_factor=0.5)
        vliser.plot_graph(all_bias_grads_var, "bias var from start to end of network", "module number", "bias var", padding_factor=0.5)
        vliser.plot_graph(all_output_var, "output var from start to end of network", "module number", "output var", padding_factor=0.5)


print("----- END OF TRAINING -----\n")

modelObserver.remove_hooks()

print(f"grad norms: {grad_norms}")
print(f"val_losses: {val_losses}")

vliser.plot_graph(grad_norms, "grad norms", "iteration", "grad norm")
vliser.plot_graph(train_losses, "training loss", "iteration", "loss")
vliser.plot_graph(val_losses, "val loss", "iteration", "loss")
print(f"Lowest val loss achieved: {lowest_val_loss}")
print(f"Lowest normalized val loss achieved: {lowest_normalized_val_loss}")
print(f"Total absolute price difference: {total_absolute_price_diff}")
