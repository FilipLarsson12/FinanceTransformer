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
