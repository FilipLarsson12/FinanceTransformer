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
def plot_heatmap(ax, data, title):
    print(f"Plotting heatmap: {title}")
    print(f"Original data shape: {data.shape}")
    # Ensure data is 2D for plotting
    if data.ndim == 1:
        data = data.reshape(1, -1)  # Convert to a single row
        print(f"Reshaped data to 2D (1 row): {data.shape}")
    elif data.ndim == 3:
        data = data.reshape(-1, data.shape[-1])
        print(f"Reshaped data from 3D to 2D: {data.shape}")
    # if 2D data is small enough print the weight values in the plot as well 
    if data.shape[0] < 10 and data.shape[1] < 10:
      sns.heatmap(data, annot=True, fmt=".3f", cmap='coolwarm', cbar=True, center=0, ax=ax)
    else:
      sns.heatmap(data, annot=False, cmap='coolwarm', cbar=True, center=0, ax=ax)

    ax.set_title(title)
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

        self.hook_handles = []

        # so I can keep track of states in all the submodules
        self.module_activities = {}

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


    # add forward and backward hooks to all submodules
    def register_hooks(self, save_grads=False):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
                forward_handle = module.register_forward_hook(lambda m, i, o, n=name: self.save_module_activity_forward(n, m, i, o))
                self.hook_handles.append(forward_handle)
                if save_grads:
                    if module is not self.position_embeddings and module is not self.price_embeddings:
                        backward_handle = module.register_full_backward_hook(lambda m, gi, go, n=name: self.save_module_activity_backward(n, m, gi, go))
                        self.hook_handles.append(backward_handle)

                    # special handling for price and pos embeddings
                    else:
                        backward_handle = module.weight.register_hook(lambda grad, n=name: self.save_grad(n, grad, 'weight grads'))
                        self.hook_handles.append(backward_handle)
                        if hasattr(module, 'bias') and module.bias is not None:
                            emb_bias_handle = module.bias.register_hook(lambda grad, n=name: self.save_grad(n, grad, 'bias grads'))
                            self.hook_handles.append(emb_bias_handle)     


    # special function to save grads for weight and bias for price and pos embeddings
    def save_grad(self, name, grad, grad_type):
        if name not in self.module_activities:
            self.module_activities[name] = {}
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
        print(f"grad input type for {module_name}: {type(grad_input)}")
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


    def plot_module_activities(self):
        for module_name, module_activity in self.module_activities.items():
            print(f"Plotting module activity for {module_name}")

            forward_keys = ['input', 'weights', 'bias', 'output']
            backward_keys = ['grad input', 'grad output', 'weight grads', 'bias grads']

            forward_activities = {k: v for k, v in module_activity.items() if k in forward_keys and v is not None}
            backward_activities = {k: v for k, v in module_activity.items() if k in backward_keys and v is not None}

            n_forward_activities = len(forward_activities)
            n_backward_activities = len(backward_activities)

            print(f"Number of forward activities: {n_forward_activities}")
            print(f"Number of backward activities: {n_backward_activities}")

            plot_width = max(n_forward_activities, n_backward_activities)

            fig, axes = plt.subplots(2, plot_width, figsize=(5 * plot_width, 10))

            if n_forward_activities > 0:
                # Plot forward activities
                for i, (key, data) in enumerate(forward_activities.items()):
                    print(f"Plotting forward activity {i+1}/{n_forward_activities}: {key}")
                    if data is not None:
                        plot_heatmap(axes[0, i], data, f'{key} for {module_name}')
                    else:
                        print("Data is None")
                for i in range(n_forward_activities, plot_width):
                    # Hide the subplot
                    axes[0, i].set_visible(False)

            if n_backward_activities > 0:
                # Plot backward activities
                for i, (key, data) in enumerate(backward_activities.items()):
                    print(f"Plotting backward activity {i+1}/{n_backward_activities}: {key}")
                    if data is not None:
                        plot_heatmap(axes[1, i], data, f'{key} for {module_name}')
                    else:
                        print("Data is None")
                for i in range(n_backward_activities, plot_width):
                    # Hide the subplot
                    axes[1, i].set_visible(False)

            plt.tight_layout()
            plt.show()


    def forward(self, idx, targets=None):
        _, T, _ = idx.size()

        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

        # input dim is: (batch_am, block_size, 1)

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.position_embeddings(pos) # position embeddings of shape (T, embd_dim)
        price_emb = self.price_embeddings(idx) # price embeddings of shape (B, T, embd_dim)

        print(f"Price embeddings shape: {price_emb.shape}")

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
    input_dim: int = 1
    embd_dim: int = 4
    block_size: int = 3
    n_layers: int = 1
    n_head: int = 1

# ----------------------------------------------------------------------------------------

#init
model_config = ModelConfig()

model = FinanceTransformer(model_config)

# get the size of the model
num_parameters = model.calculate_parameters()
print(f"Number of parameters in the model: {num_parameters}")

# define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

vliser = Visualizer()

# track losses
losses = []

# load the data into our program
dataLoader = DataLoader(B=2, T=model_config.block_size)

data_file = "data.txt"

tickerList = ['MSFT']

dataLoader.load_data_from_yfinance(tickerList, data_file, startDate='2015-01-01', endDate='2015-04-01')

print(f"dataloader.dataPerEpoch: {dataLoader.dataPerEpoch}")

model.train()

epochs = 5

batch_size = dataLoader.B * dataLoader.T

# calculate amount of batches in one epoch
total_batches = dataLoader.dataPerEpoch*epochs // batch_size

print(f"total batches: {total_batches}")

# turn on/off prints
prints = False

# register hooks so model can save state during forward and backward pass
model.register_hooks(save_grads=True)

print(f"Model config: {model_config}")

# training run

print("\n----- START OF TRAINING -----")

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

        if i == 1:
          print("\nSubmodule activities:")
          for name, activities in model.module_activities.items():
            print(f"activities for {name}")
            for activity, value in activities.items():
              print(f"{activity}: {value.shape}")
          model.plot_module_activities()
          # removing hooks
          model.remove_hooks()

        # update weights
        optimizer.step()

        # append the loss
        losses.append(loss.item())

        # Update the progress bar
        pbar.update(1)

      if prints:
        print(f"Price Embedding Layer Gradients: {model.price_embeddings.weight.grad}")
        print(f"Position Embedding Layer Gradients: {model.position_embeddings.weight.grad}")

print("----- END OF TRAINING -----\n")

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

