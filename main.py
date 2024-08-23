import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Literal
import requests
from google.colab import userdata
from dataclasses import dataclass, field
import time


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
        # nested dictionary to store min and max values by ticker and feature type
        self.tickers_min_max = {}

    def normalize(self, tickername, data, feature_type, split):
        if self.norm_scheme == "z":
            return self.z_normalize(tickername, data, split)
        elif self.norm_scheme == "percentage":
            return self.percentage_normalize(tickername, data, split)
        elif self.norm_scheme == "min_max":
            return self.min_max_normalize(tickername, data, feature_type, split)

    def de_normalize(self, tickername, normalized_data, feature_type):
        if self.norm_scheme == "z":
            return self.z_denormalize(tickername, normalized_data)
        elif self.norm_scheme == "percentage":
            return self.percentage_denormalize(tickername, normalized_data)
        elif self.norm_scheme == "min_max":
            return self.min_max_denormalize(tickername, normalized_data, feature_type)


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


    # Function to normalize using min-max scaling
    def min_max_normalize(self, tickername, data, feature_type, split="train"):
        if split == "train":
            min_val = np.min(data)
            max_val = np.max(data)
            # below is some code that may become important if i change validation set normalization but useless for now
            # initialize the dictionary for this ticker if not already done
            if tickername not in self.tickers_min_max:
                self.tickers_min_max[tickername] = {}
            # save the min and max for the ticker and feature type
            self.tickers_min_max[tickername][feature_type] = (min_val, max_val)
        elif split == "val":
            min_val = np.min(data)
            max_val = np.max(data)

            if tickername not in self.tickers_min_max:
                self.tickers_min_max[tickername] = {}
            # save the min and max for the ticker and feature type
            self.tickers_min_max[tickername][feature_type] = (min_val, max_val)

            # will uncomment if I change normalization
            '''
            if tickername not in self.tickers_min_max:
              raise ValueError(f"Training statistics for ticker {tickername} not found.")
            min_price, max_price = self.tickers_min_max[tickername]
            else:
                raise ValueError(f"Invalid split value: {split}")
            '''

        # Apply min-max normalization
        normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
        return normalized_data

    # Function to reverse the min-max normalization
    def min_max_denormalize(self, tickername, normalized_data, feature_type):
        if tickername not in self.tickers_min_max or feature_type not in self.tickers_min_max[tickername]:
            raise ValueError(f"Min and max for {feature_type} of ticker {tickername} not found.")
        min_val, max_val = self.tickers_min_max[tickername][feature_type]
        original_data = (normalized_data + 1) / 2 * (max_val - min_val) + min_val
        return original_data



class DataLoader:
    def __init__(self, model_config, normalizer):
        self.normalizer = normalizer
        self.B = model_config.batch_size
        self.T = model_config.block_size
        self.input_features = model_config.input_features

        # store train and val data
        self.train_data = {}  # Initialize train_data as empty dict
        self.val_data = {}  # Initialize val_data as empty dict

        # state to load batches of train data
        self.current_ticker = None
        self.current_position = 0

        # attribute to track the total amount of rows in train_data
        self.dataPerEpoch = 0

    # Function to download price and volume data from Yahoo Finance
    def download_yahoo_finance_data(self, ticker, start_date, end_date, split):
        # Download the price and volume data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Rename the Close column to price
        data.rename(columns={'Close': 'price'}, inplace=True)

        # Download the S&P 500 data
        sp500_data = yf.download("^GSPC", start=start_date, end=end_date)
        sp500_data.rename(columns={'Close': 's&p 500'}, inplace=True)

        # Merge the S&P 500 data with the ticker data on the Date
        data = data.merge(sp500_data[['s&p 500']], left_index=True, right_index=True)

        # Get the outstanding shares
        key_figures = yf.Ticker(ticker)
        outstanding_shares = key_figures.info['sharesOutstanding']

        # Normalize volumes
        data['volume'] = data['Volume'] / outstanding_shares

        # Return the Close price, Normalized Volume, and S&P 500 Close columns with Date
        return data[['price', 'volume', 's&p 500']].reset_index()


    # Function to fetch and process earnings data from Alpha Vantage
    def download_eps_data(self, ticker):
        url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={api_key}'
        response = requests.get(url)
        eps_data = response.json().get("quarterlyEarnings", [])
        for record in eps_data:
            for key in ['estimatedEPS', 'surprise', 'surprisePercentage', 'reportTime']:
                record.pop(key, None)

        eps_df = pd.DataFrame(eps_data)
        eps_df['reportedDate'] = pd.to_datetime(eps_df['reportedDate'])
        eps_df = eps_df.sort_values(by='reportedDate')
        return eps_df


    # Function to combine price, volume and earnings data
    def combine_price_and_eps_data(self, ticker, start_date, end_date, split):
        price_data = self.download_yahoo_finance_data(ticker, start_date, end_date, split)
        eps_data = self.download_eps_data(ticker)

        merged_data = pd.merge_asof(
            price_data,
            eps_data,
            left_on='Date',
            right_on='reportedDate',
            direction='backward'
        )

        # Rename columns
        merged_data.rename(
            columns={
                'reportedEPS': 'latest_eps',
                'fiscalDateEnding': 'eps_fiscalDate',
                'reportedDate': 'eps_reportedDate'
            },
            inplace=True
        )

        # Convert latest_eps to numeric
        merged_data['latest_eps'] = pd.to_numeric(merged_data['latest_eps'], errors='coerce')

        # Calculate inverse P/E ratio and add as a new column
        merged_data['inverse p/e'] = (merged_data['latest_eps'] * 4) / merged_data['price']

        # Normalize the specified features
        for feature in self.input_features:
            merged_data[feature] = self.normalizer.normalize(ticker, merged_data[feature].values, feature, split)

        # Determine how many rows to drop based on the split type
        if split == "train":
            pop_elements = (len(merged_data) - 1) % (self.T * self.B)
        elif split == "val":
            pop_elements = (len(merged_data) - 1) % self.T

        # Drop the last `pop_elements` rows from the DataFrame
        if pop_elements > 0:
            merged_data = merged_data[:-pop_elements]

        # Update the dataPerEpoch attribute
        if split == "train":
            self.dataPerEpoch += len(merged_data) - 1


        # Reorder columns if necessary
        columns_order = ['Date', 'price', 'volume', 'inverse p/e', 's&p 500', 'latest_eps', 'eps_fiscalDate', 'eps_reportedDate']
        merged_data = merged_data[columns_order]

        return merged_data


    # this method downloads all ticker data and return a dictionary of dataframe where each dataframe holds the data for one ticker
    def download_tickers_data(self, ticker_list, start_date, end_date, split):

        # keep track of training period to use for z-normalizing test prices later
        if split == "train":
          self.normalizer.trainStartDate = start_date
          self.normalizer.trainEndDate = end_date

        # Loop through each ticker and process data
        for ticker in ticker_list:
            merged_data = self.combine_price_and_eps_data(ticker, start_date, end_date, split)
            if split == "train":
                self.train_data[ticker] = merged_data  # Add DataFrame to train dictionary
            elif split == "val":
                self.val_data[ticker] = merged_data  # Add DataFrame to val dictionary

            print(f"Data for {ticker} written to {ticker}_{split}_data.csv")
            merged_data.to_csv(f"{ticker}_{split}_data.csv", index=False)

        # Initialize current ticker and position
        self.current_ticker = list(self.train_data.keys())[0]
        self.current_position = 0


    def get_feature_tensor(self, ticker_data, feature_name, start_idx, end_idx, split):
        """
        Helper method to fetch and transform a specific feature from the data.
        """
        feature_values = ticker_data[feature_name].values[start_idx:end_idx]
        if split == "train":
            feature_tensor = torch.Tensor(feature_values).view(self.B, self.T, 1)
        elif split == "val":
            feature_tensor = torch.Tensor(feature_values).view(-1, self.T, 1)
        return feature_tensor


    def next_batch(self):
        # Pluck out next batch
        ticker_data = self.train_data[self.current_ticker]
        batch_inputs = []

        start_idx = self.current_position
        end_idx = self.current_position + self.B * self.T

        batch_targets = ticker_data['price'].values[start_idx + 1:end_idx + 1]
        batch_targets = torch.Tensor(batch_targets).view(self.B, self.T, 1)

        # Fetch feature tensors only for features specified in config
        for feature_name in self.input_features:
            feature_tensor = self.get_feature_tensor(ticker_data, feature_name, start_idx, end_idx, split="train")
            batch_inputs.append(feature_tensor)


        # Combine features into one tensor if multiple are present
        if len(batch_inputs) > 1:
            batch_inputs = torch.cat(batch_inputs, dim=2)
        else:
            batch_inputs = batch_inputs[0]  # Only one feature, so no need to concatenate

        # Update the pointer
        self.current_position += self.B * self.T

        # If next batch is out of bounds, reset pointer
        if self.current_position + self.B * self.T + 1 > len(ticker_data):
            self.current_position = 0
            # If we are at the last ticker, go to the first ticker again
            tickers = list(self.train_data.keys())
            current_ticker_index = tickers.index(self.current_ticker)
            if current_ticker_index == len(tickers) - 1:
                self.current_ticker = tickers[0]
            else:
                self.current_ticker = tickers[current_ticker_index + 1]

        return batch_inputs, batch_targets


    # Method to get validation data for a specific ticker
    def get_validation_data(self, ticker):
        # Fetch validation data for the ticker
        if ticker not in self.val_data:
            raise ValueError(f"No validation data found for ticker {ticker}")

        ticker_data = self.val_data[ticker]
        inputs_list = []

        # Define start and end indices for slicing
        start_idx = 0
        end_idx = len(ticker_data) - 1

        # Fetch feature tensors for each input feature specified
        for feature_name in self.input_features:
            feature_tensor = self.get_feature_tensor(ticker_data, feature_name, start_idx, end_idx, split="val")
            inputs_list.append(feature_tensor)

        val_targets = ticker_data['price'].values[1:]
        targets = torch.Tensor(val_targets).view(-1, self.T, 1)

        # Combine features into one tensor if both are present
        if len(inputs_list) > 1:
            inputs = torch.cat(inputs_list, dim=2)
        else:
            inputs = inputs_list[0]  # Only one feature, so no need to concatenate

        return inputs, targets



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
        self.num_input_features = len(config.input_features)

        # input embeddings
        self.price_embeddings = nn.Linear(config.input_dim, config.embd_dim)
        self.volume_embeddings = nn.Linear(config.input_dim, config.embd_dim)
        self.inverse_pe_embeddings = nn.Linear(config.input_dim, config.embd_dim)
        self.sp_500_embeddings = nn.Linear(config.input_dim, config.embd_dim)

        self.position_embeddings = nn.Embedding(config.block_size, config.embd_dim * self.num_input_features)

        # layers
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_normlayer = nn.LayerNorm(config.embd_dim * self.num_input_features)
        self.predictionlayer = nn.Linear(config.embd_dim * self.num_input_features, 1)

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
        # idx: (batch_size, block_size, F) where F represents the number of features

        _, T, F = idx.size()

        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        assert F == len(self.config.input_features), f"Expected {len(self.config.input_features)} input features, but got {F}"

        # Initialize an empty list to hold the embeddings
        embeddings = []

        # Helper function to embed a feature
        def embed_feature(feature_name, embedding_layer):
            if feature_name in self.config.input_features:
                feature_index = self.config.input_features.index(feature_name)
                feature_data = idx[:, :, feature_index].unsqueeze(-1)  # (batch_size, block_size, 1)
                feature_emb = embedding_layer(feature_data)  # (batch_size, block_size, embd_dim)
                embeddings.append(feature_emb)

        # Embed each feature
        embed_feature("price", self.price_embeddings)
        embed_feature("volume", self.volume_embeddings)
        embed_feature("inverse p/e", self.inverse_pe_embeddings)
        embed_feature("s&p 500", self.sp_500_embeddings)

        # Concatenate all embeddings along the last dimension
        x = torch.cat(embeddings, dim=-1)  # (batch_size, block_size, embd_dim * num_features)

        # Generate positional encodings for the sequence length T
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # Positional encoding
        pos_emb = self.position_embeddings(pos)

        # Add positional encodings to the concatenated embeddings
        x = x + pos_emb  # (batch_size, block_size, embd_dim * num_features)

        # Pass the concatenated embeddings through the transformer blocks
        for i, block in enumerate(self.layers):
            x = block(x)  # (batch_size, block_size, embd_dim * num_features)

        # Apply the final layer normalization
        x = self.final_normlayer(x)  # (batch_size, block_size, embd_dim * num_features)

        # Predict and compute loss if targets are provided
        if targets is not None:
            preds = self.predictionlayer(x)  # (batch_size, block_size, 1)
            if self.config.loss_fn == "L1":
                loss_fn = nn.L1Loss()
            elif self.config.loss_fn == "MSE":
                loss_fn = nn.MSELoss()
            loss = loss_fn(preds, targets)
        else:
            preds = self.predictionlayer(x)  # (batch_size, block_size, 1)
            loss = None

        return preds, loss




# class that defines the self attention layers
class SelfAttentionLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # we use a linear layer to produce the k, q and v.
        # we use the same dimensionality for them as the embd_dim * num_input_features
        self.num_input_features = len(config.input_features)
        self.combined_embd_dim = config.embd_dim * self.num_input_features  # updated to account for concatenated embeddings
        self.n_head = config.n_head

        # must make sure that we can use several attention heads
        assert self.combined_embd_dim % config.n_head == 0, f"Embedding dimension {config.embd_dim} is not divisible by the number of heads {config.n_head}."

        # to calculate k, q and v
        self.calc_k = nn.Linear(self.combined_embd_dim, self.combined_embd_dim)
        self.calc_q = nn.Linear(self.combined_embd_dim, self.combined_embd_dim)
        self.calc_v = nn.Linear(self.combined_embd_dim, self.combined_embd_dim)

        # final proj before return
        self.proj = nn.Linear(self.combined_embd_dim, self.combined_embd_dim)
        # flag for weight scaling to counteract residual streams increasing variance in the forward pass
        self.proj.RESIDUAL_SCALING_INIT = 1

        # register parameter for the lower triangular mask-matrix
        self.register_buffer("bias",
        torch.tril(torch.ones(config.block_size, config.block_size))
        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, combined_embd_dim

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
        keyquery_matrix = keyquery_matrix.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

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
        self.num_input_features = len(config.input_features)
        self.combined_embd_dim = config.embd_dim * self.num_input_features  # updated to account for concatenated embeddings

        self.proj_1 = nn.Linear(self.combined_embd_dim, 4 * self.combined_embd_dim)
        self.act_fn = nn.GELU(approximate='tanh')
        #self.act_fn = nn.LeakyReLU()
        self.proj_2 = nn.Linear(self.combined_embd_dim * 4, self.combined_embd_dim)
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
        self.num_input_features = len(config.input_features)
        self.combined_embd_dim = config.embd_dim * self.num_input_features  # updated to account for concatenated embeddings
        self.ln_1 = nn.LayerNorm(self.combined_embd_dim)
        self.attn = SelfAttentionLayer(config)
        self.ln_2 = nn.LayerNorm(self.combined_embd_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# model config class
@dataclass
class ModelConfig():
    model_name: str
    input_dim: int = 1
    input_features: list = field(default_factory=lambda: ["price", "volume", "inverse p/e", "s&p 500"])
    batch_size: int = 4
    block_size: int = 64  # Reduce block size to focus on shorter sequences
    embd_dim: int = 64  # Reduce embedding dimension to simplify the model
    n_layers: int = 16  # Reduce the number of layers for shallower learning
    n_head: int = 8  # Reduce the number of attention heads for simpler attention mechanism
    loss_fn: Literal['L1', 'MSE'] = "MSE"

def config_to_tuple(config):
    # Convert the ModelConfig object to a tuple, converting lists to tuples
    return (
        config.input_dim,
        tuple(config.input_features),  # Convert list to tuple
        config.batch_size,
        config.block_size,
        config.embd_dim,
        config.n_layers,
        config.n_head,
        config.loss_fn
    )

# ----------------------------------------------------------------------------------------

# autodetect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.set_float32_matmul_precision('high')

# init models
model_config_price = ModelConfig(model_name="Model_Price", input_features=["price"])
model_config_price_volume = ModelConfig(model_name="Model_Price_Volume", input_features=["price", "volume"])
model_config_price_volume_inverse_pe = ModelConfig(model_name="Model_Price_Volume_Inverse_PE", input_features=["price", "volume", "inverse p/e"])
model_config_price_volume_inverse_pe_sp_500 = ModelConfig(model_name="Model_Price_Volume_Inverse_PE_SP500", input_features=["price", "volume", "inverse p/e", "s&p 500"])


model_list = []

model_price = FinanceTransformer(model_config_price)
model_price_volume = FinanceTransformer(model_config_price_volume)
model_price_volume_inverse_pe = FinanceTransformer(model_config_price_volume_inverse_pe)
model_price_volume_inverse_pe_sp_500 = FinanceTransformer(model_config_price_volume_inverse_pe_sp_500)


model_list.append(model_price)
model_list.append(model_price_volume)
model_list.append(model_price_volume_inverse_pe)
model_list.append(model_price_volume_inverse_pe_sp_500)

# used to save stats for each model for comparison
model_stats = {}


# Updated training and validation sets
train_ticker_list = ['TSLA', 'CVX', 'UNH', 'NKE', 'GOOGL', 'JPM', 'KO']
train_start_date = "2018-01-01"
train_end_date = "2023-01-01"

val_ticker_list = ['META', 'X', 'NFLX', 'CVS', 'ORCL', 'BA', 'DIS', 'MMM', 'CAT', 'ADBE']
val_start_date = "2023-01-01"
val_end_date = "2024-04-01"

# api key for fetching data from Alpha Vantage
api_key = userdata.get('ALPHA_VANTAGE_API_KEY')

# used for visualizing model activity and model output
vliser = Visualizer()


for model in model_list:
  model.to(device)

  # object to track model activities
  modelObserver = ModelObserver(model)

  # get the size of the model
  num_parameters = model.calculate_parameters()
  print(f"Number of parameters in the model: {num_parameters}")

  # define loss function and optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

  # track losses and grad norms
  train_losses = []
  val_losses = []
  # used so I can compare losses between different normalization schemes
  normalized_val_losses = []
  # used to inspect training stability
  grad_norms = []

  # Instantiate the DataNormalizer class
  normalizer = DataNormalizer(norm_scheme="min_max")

  # Instantiate the DataLoader class
  data_loader = DataLoader(model.config, normalizer)

  # Process the ticker list
  data_loader.download_tickers_data(train_ticker_list, train_start_date, train_end_date, split="train")

  data_loader.download_tickers_data(val_ticker_list, val_start_date, val_end_date, split="val")


  # how many batches between each val loss measurement
  val_interval = 25
  # record lowest val loss
  lowest_val_loss = float('inf')
  lowest_normalized_val_loss = float('inf')
  total_absolute_price_diff = 0

  print(f"dataloader.dataPerEpoch: {data_loader.dataPerEpoch}")

  epochs = 40

  batch_size = data_loader.B * data_loader.T

  # calculate amount of batches in one epoch
  total_batches = data_loader.dataPerEpoch*epochs // batch_size

  print(f"total batches: {total_batches}")

  # toggle prints, plots and hooks
  prints = False
  plots = False
  hooks = False


  # register hooks so modelObserver can save model state during forward and backward pass
  if hooks:
    modelObserver.register_hooks(save_grads=True)

  print(f"Model config: {model.config}")

  # training run

  print("\n----- START OF TRAINING -----")

  with tqdm(total=(total_batches), desc=f"Training progress") as pbar:

    for i in range(total_batches):

        # set the model to train mode
        model.train()

        t0 = time.time()

        # load batch
        X, Y = data_loader.next_batch()

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

          if device.type == 'cuda':
              torch.cuda.synchronize()

          t1 = time.time()
          dt = (t1 - t0)
          prices_per_sec = (data_loader.B * data_loader.T) / dt

          if i % 25 == 0:
            print(f"batch {i}, prices/second {prices_per_sec}")

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

          for ticker in data_loader.val_data.keys():

            # Fetch validation inputs and targets using the new method
            inputs, targets = data_loader.get_validation_data(ticker)

            # Move data to the appropriate device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Keep track of the variance of the data
            data_variance = torch.var(targets, unbiased=False).item()

            # Perform model inference
            with torch.no_grad():
                preds, val_loss_ticker = model(inputs, targets)

            # add to the overall loss
            val_loss += val_loss_ticker
            normalized_val_loss += (val_loss_ticker / data_variance)


            # plot tickers if we are at the last batch
            if (i + val_interval) >= total_batches:
              # reshape to 1D for plot
              preds = preds.view(-1).cpu().numpy()
              targets = targets.view(-1).cpu().numpy()

              deNormalize_prices = True
              if deNormalize_prices:
                # call the dataloaders denormalize() function to reverse normalization for model preds and targets of current ticker
                preds = normalizer.de_normalize(ticker, preds, "price")
                targets = normalizer.de_normalize(ticker, targets, "price")

                element_wise_diff = preds - targets
                absolute_diff = np.abs(element_wise_diff)
                sum_absolute_diff = np.sum(absolute_diff)
                total_absolute_price_diff += sum_absolute_diff

              # vliser.plot_module_activities(modelObserver)
              vliser.plot_preds(actual_prices=targets, preds=preds, title=f"Val Ticker: {ticker}, period: {val_start_date} to {val_end_date}, processed {int((i / total_batches)*100)}% of training data, inputs: {model.config.input_features}", width=16)


          # calculate average loss over tickers
          val_loss = val_loss / len(data_loader.val_data)
          normalized_val_loss = normalized_val_loss / len(data_loader.val_data)
          # append the test loss
          val_losses.append(val_loss.item())
          normalized_val_losses.append(normalized_val_loss.item())


          # check if this is lowest loss
          if val_loss.item() < lowest_val_loss:
            lowest_val_loss = val_loss
          if normalized_val_loss.item() < lowest_normalized_val_loss:
            lowest_normalized_val_loss = normalized_val_loss

  print("----- END OF TRAINING -----\n")

  if hooks:
    modelObserver.remove_hooks()

  # Store statistics for this model using its configuration as a key
  model_stats[model.config.model_name] = {
      "grad_norms": grad_norms,
      "train_losses": train_losses,
      "val_losses": val_losses,
      "normalized_val_losses": normalized_val_losses,
      "lowest_val_loss": lowest_val_loss,
      "lowest_normalized_val_loss": lowest_normalized_val_loss,
      "total_absolute_price_diff": total_absolute_price_diff
  }

# Print model statistics
for model_name, stats in model_stats.items():
    print(f"Model Name: {model_name}")

    vliser.plot_graph(stats["val_losses"], f"val loss for {model_name}", "iteration", "loss")

    print(f"Lowest normalized val loss achieved for {model_name}: {stats['lowest_normalized_val_loss']}")
    # vliser.plot_graph(stats['grad_norms'], f"grad norms for {model_name}", "iteration", "grad norm")
    total_absolute_price_diff = stats['total_absolute_price_diff']
    print(f"Total absolute price difference for {model_name}: {total_absolute_price_diff}")
