import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import tensorflow_addons as tfa
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import Precision, Recall
from keras_tuner import Hyperband, Objective
import joblib  # For saving/loading models
import os

class BalancedBatchGenerator(Sequence):
    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indices_class_0 = np.where(y == 0)[0]  # Indices for the majority class (e.g., `no-buy`)
        self.indices_class_1 = np.where(y == 1)[0]  # Indices for the minority class (e.g., `buy`)
        
        # Calculate number of samples per class per batch
        self.samples_per_class = batch_size // 2  # Half batch for each class

    def __len__(self):
        # Return the number of batches per epoch
        return int(np.floor(len(self.y) / self.batch_size))

    def __getitem__(self, idx):
        # Randomly sample indices for each class
        class_0_indices = np.random.choice(self.indices_class_0, self.samples_per_class, replace=True)
        class_1_indices = np.random.choice(self.indices_class_1, self.samples_per_class, replace=True)
        
        # Combine the indices and shuffle them
        batch_indices = np.concatenate([class_0_indices, class_1_indices])
        np.random.shuffle(batch_indices)
        
        # Get the corresponding samples
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        
        return X_batch, y_batch

def download_stock_data(ticker_list, period='max'):
    stock_data = {}
    for ticker in ticker_list:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval='1d')
        stock_data[ticker] = data
        print(f"Downloaded data for {ticker}")
    return stock_data

def calculate_rsi(df, window=14):
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    short_ema = df['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df

def add_features(df):
    df['MA_50'] = df['Close'].rolling(window=50).mean()  # 50-period moving average
    df['Volatility'] = df['Close'].rolling(window=50).std()  # Volatility (standard deviation)
    df['Momentum'] = df['Close'].diff(3)  # 3-period momentum
    df = calculate_rsi(df)  # Calculate RSI with default window of 14
    df = calculate_macd(df)  # Calculate MACD with default windows (12, 26, 9)
    
    return df.dropna()

def preprocess_data(df):
    # Check if there is enough data to process
    if df.empty or df.shape[0] < 60:  # Ensure enough rows for a 60-period lookback
        raise ValueError("Not enough data to process.")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Ensure required columns are present and contain enough data
    try:
        scaled_data = scaler.fit_transform(df[['Close', 'MA_50', 'Volatility', 'Momentum', 'RSI', 'MACD', 'MACD_Signal']])
    except ValueError as e:
        print(f"Error scaling data: {e}")
        raise
    
    X = []
    y = []
    
    lookback = 60
    for i in range(lookback, len(scaled_data) - 30):
        # Get the starting price
        starting_price = scaled_data[i][0]
        
        # Skip if starting price is 0 to avoid division by zero
        if starting_price == 0:
            continue
        
        # Check if the stock rises by 7.5% in 30 days
        price_in_30_days = scaled_data[i+30][0]
        max_drop = np.min(scaled_data[i:i+30, 0])  # Minimum price within the 30 days
        
        # Condition 1: Stock rises by at least 7.5%
        rise_condition = (price_in_30_days - starting_price) / starting_price >= 0.075
        
        # Condition 2: Stock does not drop more than 1% below the starting price
        drop_condition = (max_drop - starting_price) / starting_price >= -0.01
        
        # Only append 1 (buy) if both conditions are met, otherwise append 0
        if rise_condition and drop_condition:
            X.append(scaled_data[i-lookback:i])
            y.append(1)  # Ensure labels are integers
        else:
            X.append(scaled_data[i-lookback:i])
            y.append(0)  # Ensure labels are integers
    
    if len(X) == 0 or len(y) == 0:  # If no valid data was found, skip
        raise ValueError("No valid data after preprocessing.")

    X = np.array(X)
    y = np.array(y, dtype=int)  # Explicitly cast the labels as integers
    
    return X, y, scaler

def build_model_with_tuner(hp):
    model = Sequential()
    model.add(Input(shape=(60, 7)))  # Adjust according to your input shape

    # LSTM layer with tunable units
    model.add(Bidirectional(LSTM(
        units=hp.Int('lstm_units_1', min_value=32, max_value=128, step=32),
        return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))

    # Second LSTM layer with tunable units
    model.add(LSTM(units=hp.Int('lstm_units_2', min_value=32, max_value=128, step=32), return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))

    # Dense layer with tunable units
    model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=16)))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model with a tunable learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])),
        loss=tfa.losses.SigmoidFocalCrossEntropy(),
        metrics=['accuracy', Precision(), Recall()]
    )
    return model

def train_model(stock_data, model_filename='stock_model.h5', scaler_filename='scaler.pkl'):
    preprocessed_data = {}

    for ticker, data in stock_data.items():
        try:
            # Add features and preprocess data
            data = add_features(data)
            X, y, scaler = preprocess_data(data)  # Will raise ValueError if no valid data
            preprocessed_data[ticker] = (X, y, scaler)

            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Create balanced batch generator for the training data
            balanced_batch_generator = BalancedBatchGenerator(X_train, y_train, batch_size=64)

            # Initialize Keras Tuner
            tuner = Hyperband(
                build_model_with_tuner,  # The model-building function with hyperparameters
                objective=Objective("val_recall", direction="max"),  # Specify direction explicitly
                max_epochs=50,
                factor=3,
                directory='hyperparameter_tuning',
                project_name=f'stock_predictor_tuning_{ticker}'
            )

            # Early Stopping and Learning Rate Scheduler
            early_stopping = EarlyStopping(
                monitor='val_recall',  # Monitor validation recall for stopping
                patience=5,
                restore_best_weights=True,
                mode='max'
            )
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_recall',
                factor=0.3,
                patience=3,
                min_lr=1e-5,
                mode='max'
            )

            # Run hyperparameter tuning with the tuner
            tuner.search(
                balanced_batch_generator,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, lr_scheduler]
            )

            # Retrieve the best model after tuning
            best_model = tuner.get_best_models(num_models=1)[0]

            # Save the best model and scaler
            best_model.save(f"{ticker}_{model_filename}")
            joblib.dump(scaler, f"{ticker}_{scaler_filename}")
            print(f"Best model and scaler for {ticker} saved.")

        except ValueError as e:
            # Log the error and skip this ticker
            print(f"Skipping {ticker} due to error: {e}")
            continue

    return preprocessed_data

def load_model_and_scaler(ticker, model_filename='stock_model.h5', scaler_filename='scaler.pkl'):
    model = tf.keras.models.load_model(f"{ticker}_{model_filename}")
    scaler = joblib.load(f"{ticker}_{scaler_filename}")
    
    # Recompile the model with the same loss and metrics used during training
    model.compile(optimizer='adam', loss=tfa.losses.SigmoidFocalCrossEntropy(), metrics=['accuracy', Precision(), Recall()])
    
    return model, scaler

def generate_signal_with_confidence(model, data, scaler):
    # Ensure data is not empty
    if data.empty or data.shape[0] < 60:
        print("Not enough data to make predictions.")
        return []  # Return an empty list if there's no data

    try:
        # Apply the scaler on relevant columns
        scaled_data = scaler.transform(data[['Close', 'MA_50', 'Volatility', 'Momentum']])
    except ValueError as e:
        print(f"Error scaling data: {e}")
        return []  # Return an empty list if scaling fails

    lookback = 60
    X = []
    
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
    
    X = np.array(X)
    
    # Ensure we have data to make predictions
    if len(X) == 0:
        print("No valid data for prediction.")
        return []  # Return an empty list if no valid data
    
    predictions = model.predict(X)

    buy_signals = []
    
    for i in range(len(predictions) - 30):
        # Get the predicted closing prices for the next 30 days
        starting_price = scaled_data[i][0]
        future_prices = scaled_data[i:i+30, 0]
        price_in_30_days = future_prices[-1]
        max_drop = np.min(future_prices)  # Minimum price in the next 30 days
        
        # Condition 1: Stock rises by at least 7.5%
        rise_condition = (price_in_30_days - starting_price) / starting_price >= 0.075
        
        # Condition 2: Stock does not drop more than 1% below the starting price
        drop_condition = (max_drop - starting_price) / starting_price >= -0.01
        
        # Only append a buy signal if both conditions are met
        if rise_condition and drop_condition:
            buy_signals.append((i, predictions[i][0]))  # Store the index and confidence of the buy signal
    
    return buy_signals

def fetch_real_time_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1y', interval='1d')
        
        # Ensure data has at least 60 rows to process
        if data.empty or data.shape[0] < 60:
            raise ValueError(f"Not enough data available for ticker {ticker}")
        
        data = add_features(data)
        
        # Ensure data is still valid after feature engineering
        if data.empty:
            raise ValueError(f"Not enough valid data after feature engineering for ticker {ticker}")
        
        return data.dropna()
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None  # Return None if there's an error

def main():
    # List of S&P 500 stock tickers
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)[0]

    # Extract the tickers
    sp500_tickers = table['Symbol'].to_list()

    # Ask user if they want to download new data and train the model
    user_choice = input("Would you like to download new stock data and train a new model? (yes/no): ").lower()

    if user_choice == 'yes':
        stock_data = download_stock_data(sp500_tickers)
        for ticker in stock_data:
            stock_data[ticker].to_csv(f"{ticker}_data.csv")
            print(f"Data for {ticker} saved to file.")
        train_model(stock_data)
    else:
        buy_signals = []
                
        # Go through each ticker and generate predictions
        for ticker in sp500_tickers:
            # Check if saved model exists for the stock
            if os.path.exists(f"{ticker}_stock_model.h5"):
                model, scaler = load_model_and_scaler(ticker)
                real_time_data = fetch_real_time_data(ticker)
                
                if real_time_data is None:
                    print(f"Skipping {ticker} due to unavailable real-time data.")
                    continue
                
                predictions = generate_signal_with_confidence(model, real_time_data, scaler)
                
                # Check the latest prediction
                if predictions:
                    latest_prediction = predictions[-1][1]  # Get the latest confidence score
                    
                    # If it's a buy signal (confidence >= 0.5), add to the list
                    if latest_prediction >= 0.5:
                        buy_signals.append((ticker, latest_prediction))
            else:
                print(f"No saved model found for {ticker}. Skipping...")

        # Sort buy signals by model confidence in descending order
        buy_signals = sorted(buy_signals, key=lambda x: x[1], reverse=True)
        
        # Print the tickers with buy signals and their confidence
        if buy_signals:
            print("\nTickers with Buy signals, ordered by confidence:")
            for ticker, confidence in buy_signals:
                print(f"{ticker}: Buy signal with {confidence * 100:.2f}% confidence")
        else:
            print("No Buy signals found.")

if __name__ == '__main__':
    main()