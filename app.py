# from polygon import RESTClient

# client = RESTClient(api_key="ppisGP51Q3ZPNYFSX8hCdWleFNzKa5I0")

# ticker = "JMIA"

import pandas as pd
import requests
import time
import numpy as np
import logging
from logging.handlers import RotatingFileHandler

# Constants
API_KEY = 'ppisGP51Q3ZPNYFSX8hCdWleFNzKa5I0'
BASE_URL_AGG = 'https://api.polygon.io/v2/aggs/ticker/'  # For closing prices
BASE_URL_SMA = 'https://api.polygon.io/v1/indicators/sma/'  # For SMA
BASE_URL_AGG = 'https://api.polygon.io/v2/aggs/ticker/'  # For historical data
CALL_LIMIT = 2  # API calls per minute
DELAY = 60 / CALL_LIMIT  # Delay to respect API rate limit

# Set up logger
def setup_logger():
    logger = logging.getLogger('polygon_api_logger')
    logger.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create file handler
    file_handler = RotatingFileHandler('polygon_api.log', maxBytes=1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Global logger
logger = setup_logger()


# Read the tickers from a CSV file
def load_tickers(input_csv):
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} tickers from {input_csv}")
    return df['Ticker'].tolist()

# Fetch closing prices for a ticker from Polygon API
def get_closing_prices(ticker):
    url = f"{BASE_URL_AGG}{ticker}/prev?apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        logger.info(f"Successfully fetched closing prices for {ticker}")
        if 'results' in data:
            return [result['c'] for result in data['results']]  # Extract closing prices
    else:
        logger.error(f"Error fetching data for {ticker}: {response.status_code}")
    return []

# Fetch 200-day SMA for a ticker from Polygon API
def get_200_day_sma(ticker):
    url = f"{BASE_URL_SMA}{ticker}?timespan=day&adjusted=true&window=200&series_type=close&order=desc&apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Check for valid response structure
        if 'results' in data and 'values' in data['results'] and len(data['results']['values']) > 0:
            return data['results']['values'][0]['value']  # Extract the latest SMA value
    else:
        print(f"Error fetching 200-day SMA for {ticker}: {response.status_code}")
        return None

# Fetch historical data for a ticker from Polygon API# Fetch historical data for a ticker from Polygon API
def get_historical_data(ticker, days=15):
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    url = f"{BASE_URL_AGG}{ticker}/range/1/day/{start_date}/{end_date}?apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'results' in data:
            logger.info(f"Successfully fetched historical data for {ticker}")
            return pd.DataFrame(data['results'])
    else:
        logger.error(f"Error fetching historical data for {ticker}: {response.status_code}")
    return pd.DataFrame()

# Calculate RSI# Calculate RSI
def calculate_rsi(data, periods=14):
    if len(data) < periods + 1:
        logger.warning(f"Not enough data to calculate RSI. Need at least {periods + 1} data points.")
        return np.nan
    
    close_diff = data['c'].diff()
    
    # Separate gains and losses
    gains = close_diff.where(close_diff > 0, 0)
    losses = -close_diff.where(close_diff < 0, 0)
    
    # Calculate average gains and losses
    avg_gain = gains.rolling(window=periods, min_periods=periods).mean()
    avg_loss = losses.rolling(window=periods, min_periods=periods).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1]


# Calculate the percentage gap between closing price and SMA
def calculate_gap(closing_price, sma):
    if sma is None:
        return None  # If there's no SMA, gap can't be calculated
    return ((closing_price - sma) / sma) * 100


# Save results to CSV
def save_to_csv(output_csv, results):
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    logger.info(f"Results saved to {output_csv}")

def determine_action(gap, rsi):
    if pd.isna(gap) or pd.isna(rsi):
        return "Insufficient Data"
    
    if gap > 0:
        if rsi < 30:
            return "Buy Call"
        elif rsi > 70:
            return "Monitor"
        else:
            return "No Action"
    else:  # gap <= 0
        if rsi > 70:
            return "Buy Put"
        elif rsi < 30:
            return "Might buy call"
        else:
            return "No Action"

def main(input_csv, output_csv):
    tickers = load_tickers(input_csv)
    results = []

    for ticker in tickers:
        logger.info(f"Processing ticker: {ticker}")
        try:
            historical_data = get_historical_data(ticker, days=30)
            if not historical_data.empty:
                closing_price = historical_data['c'].iloc[-1]
                sma = get_200_day_sma(ticker)
                gap = calculate_gap(closing_price, sma)
                rsi = calculate_rsi(historical_data)
                action = determine_action(gap, rsi)
                
                result = {
                    'Ticker': ticker, 
                    'Closing Price': closing_price, 
                    '200 SMA': sma, 
                    'Gap': gap,
                    'RSI': rsi,
                    'Action': action
                }
                results.append(result)
                
                # Use a safer logging approach
                log_message = f"Processed {ticker}: Closing Price={closing_price}"
                if sma is not None:
                    log_message += f", 200 SMA={sma}"
                if gap is not None:
                    log_message += f", Gap={gap:.2f}"
                if rsi is not None:
                    log_message += f", RSI={rsi:.2f}"
                log_message += f", Action={action}"
                
                logger.info(log_message)
            else:
                logger.warning(f"No historical data available for {ticker}")
                results.append({'Ticker': ticker, 'Error': 'No historical data'})
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            results.append({'Ticker': ticker, 'Error': str(e)})
        
        # Respect rate limit
        time.sleep(DELAY)

    save_to_csv(output_csv, results)
    logger.info(f"Results saved to {output_csv}")

# Run the program
if __name__ == "__main__":
    logger.info("Starting the program")
    main('input_tickers.csv', 'output_data.csv')
    logger.info("Program completed")