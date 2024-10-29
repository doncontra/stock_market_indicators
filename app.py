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
    
    # First average
    avg_gain = gains.iloc[:periods].mean()
    avg_loss = losses.iloc[:periods].mean()
    
    # Initialize the arrays
    avg_gains = np.zeros(len(data))
    avg_losses = np.zeros(len(data))
    avg_gains[periods - 1] = avg_gain
    avg_losses[periods - 1] = avg_loss
    
    # Calculate smoothed averages
    for i in range(periods, len(data)):
        avg_gain = (avg_gains[i-1] * 13 + gains.iloc[i]) / 14
        avg_loss = (avg_losses[i-1] * 13 + losses.iloc[i]) / 14
        avg_gains[i] = avg_gain
        avg_losses[i] = avg_loss
    
    # Calculate RS and RSI
    rs = np.where(avg_losses != 0, avg_gains / avg_losses, 100.0)  # Handle division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # If there are no losses, RSI should be 100
    rsi = np.where(avg_losses == 0, 100.0, rsi)
    
    return rsi[-1]

# Calculate the percentage gap between closing price and SMA
def calculate_gap(closing_price, sma):
    if sma is None:
        return None  # If there's no SMA, gap can't be calculated
    return ((closing_price - sma) / sma) * 100


# Save results to CSV
def save_to_csv(output_csv, results):
    df = pd.DataFrame(results)
    
    # Ensure consistent column order
    columns = [
        'Ticker',
        'Closing Price',
        '200 SMA',
        'Gap',
        'RSI',
        'MACD',
        'MACD Signal',
        'Analysis'  # This will be our last column
    ]
    
    # Reorder columns and handle any missing columns
    existing_columns = [col for col in columns if col in df.columns]
    df = df[existing_columns]
    
    # Save to CSV with proper encoding for special characters
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"Results saved to {output_csv}")

def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    # Calculate the exponential moving averages
    short_ema = data['c'].ewm(span=short_period, adjust=False).mean()
    long_ema = data['c'].ewm(span=long_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = short_ema - long_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate MACD histogram
    macd_histogram = macd_line - signal_line
    
    return {
        'macd': macd_line.iloc[-1],
        'signal': signal_line.iloc[-1],
        'histogram': macd_histogram.iloc[-1]
    }

def determine_action(gap, rsi, macd_data):
    if pd.isna(gap) or pd.isna(rsi):
        return "Insufficient Data"
    
    action = "No Action"
    reasons = []
    warnings = []
    strength = []
    
    # Gap Analysis
    if gap > 20:
        warnings.append(f"Stock significantly above 200 SMA (Gap: {gap:.1f}%)")
    elif gap > 0:
        strength.append(f"Trading above 200 SMA (Gap: {gap:.1f}%)")
    else:
        warnings.append(f"Trading below 200 SMA (Gap: {gap:.1f}%)")

    # RSI Analysis
    if rsi > 80:
        warnings.append(f"Extremely overbought (RSI: {rsi:.1f})")
    elif rsi > 70:
        warnings.append(f"Overbought (RSI: {rsi:.1f})")
    elif rsi < 20:
        strength.append(f"Extremely oversold (RSI: {rsi:.1f})")
    elif rsi < 30:
        strength.append(f"Oversold (RSI: {rsi:.1f})")
    else:
        strength.append(f"RSI in neutral zone ({rsi:.1f})")

    # MACD Analysis
    macd_trend = ""
    if macd_data['macd'] > macd_data['signal']:
        macd_trend = "bullish"
        strength.append("MACD above signal line")
    else:
        macd_trend = "bearish"
        warnings.append("MACD below signal line")

    # Determine Action based on combined indicators
    if gap > 0:
        if rsi < 30:
            action = "Buy Call"
            reasons.append("Primary: RSI indicates oversold condition")
            reasons.append("Supporting: Positive gap from 200 SMA")
            if macd_trend == "bullish":
                reasons.append("Confirming: MACD shows bullish momentum")
        elif rsi > 70:
            action = "Monitor"
            reasons.append("Caution: RSI indicates overbought condition")
            reasons.append("Risk: Extended gap above 200 SMA")
        else:
            action = "No Action"
            reasons.append("RSI in neutral zone")
    else:  # gap <= 0
        if rsi > 70:
            action = "Buy Put"
            reasons.append("Primary: RSI indicates overbought condition")
            reasons.append("Supporting: Trading below 200 SMA")
            if macd_trend == "bearish":
                reasons.append("Confirming: MACD shows bearish momentum")
        elif rsi < 30:
            if macd_trend == "bullish":
                action = "Buy Call"
                reasons.append("Primary: RSI indicates oversold condition")
                reasons.append("Supporting: MACD shows bullish momentum")
                reasons.append("Warning: Trading below 200 SMA")
            else:
                action = "Monitor"
                reasons.append("Mixed signals: Oversold but bearish trend")
        else:
            action = "No Action"
            reasons.append("No clear signal")

    # Format the analysis as a single line for CSV
    analysis = f"{action}\n\nReasons:\n- {'\n- '.join(reasons)}"
    if strength:
        analysis += f"\nStrength Indicators:\n- {'\n- '.join(strength)}"
    if warnings:
        analysis += f"\nWarnings:\n- {'\n- '.join(warnings)}"

    return analysis

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
                macd_data = calculate_macd(historical_data)
                
                analysis = determine_action(gap, rsi, macd_data)
                
                result = {
                    'Ticker': ticker, 
                    'Closing Price': closing_price, 
                    '200 SMA': sma, 
                    'Gap': gap,
                    'RSI': rsi,
                    'MACD': macd_data['macd'],
                    'MACD Signal': macd_data['signal'],
                    'Analysis': analysis
                }
                
                results.append(result)
                logger.info(f"Processed {ticker}")
                logger.debug(f"Analysis for {ticker}:\n{analysis}")
            else:
                logger.warning(f"No historical data available for {ticker}")
                results.append({
                    'Ticker': ticker, 
                    'Error': 'No historical data',
                    'Analysis': 'Insufficient Data'
                })
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            results.append({
                'Ticker': ticker, 
                'Error': str(e),
                'Analysis': 'Error in Processing'
            })
        
        time.sleep(DELAY)

    save_to_csv(output_csv, results)
    logger.info(f"Results saved to {output_csv}")

# Run the program
if __name__ == "__main__":
    logger.info("Starting the program")
    main('input_tickers.csv', 'output_data.csv')
    logger.info("Program completed")