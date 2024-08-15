import pandas as pd
import numpy as np
import yfinance


def calculate_rsi(prices: pd.Series, window=14):
    delta = prices.diff()
    
    avg_gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    avg_loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean().abs()
    
    rsi = 100 * avg_gain / (avg_gain + avg_loss)
    rsi = rsi.fillna(50).to_numpy()
    
    return rsi

def calculate_macd(prices: pd.Series, long_term=26, short_term=12, signal_threshold=9):
    if len(prices) < long_term:
        raise ValueError("Not enough data points to calculate MACD.")
    
    short_term_ema = prices.ewm(span=short_term, adjust=False, min_periods=1).mean()
    long_term_ema = prices.ewm(span=long_term, adjust=False, min_periods=1).mean()
    
    macd = short_term_ema - long_term_ema
    signal_line = macd.ewm(span=signal_threshold, adjust=False, min_periods=1).mean()
    
    return macd.to_numpy(), signal_line.to_numpy()

def calculate_log_returns(prices: pd.Series):
    return np.log(prices / prices.shift(1)).dropna().to_numpy()

def calculate_volatility(prices: pd.Series, window=30):
    log_returns = calculate_log_returns(prices)
    volatility = pd.Series(log_returns).rolling(window=window).std()
    return volatility.dropna().to_numpy()

def preprocess_data(prices: pd.Series):
    prices = pd.Series(prices)
    rsi = calculate_rsi(prices)
    macd, signal = calculate_macd(prices)
    log_returns = calculate_log_returns(prices)
    volatility = calculate_volatility(prices)

    min_length = min(len(rsi), len(macd), len(signal), len(log_returns), len(volatility))
    
    prices = prices[-min_length:].to_numpy()
    rsi = rsi[-min_length:]
    macd = macd[-min_length:]
    signal = signal[-min_length:]
    log_returns = log_returns[-min_length:]
    volatility = volatility[-min_length:]

    features = np.column_stack((prices, rsi, macd, signal, log_returns, volatility))

    return features

def normalize_features(features: np.ndarray):
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)

def get_data(futures_symbol: str, start_date: str, end_date: str):
    df = yfinance.download(futures_symbol, start=start_date, end=end_date, interval="1d")
    prices = df["Close"].dropna()
    return prices

def get_data_from_csv(file_path):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df["Close"].dropna()
