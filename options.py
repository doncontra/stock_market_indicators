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


def calculate_rsi(ticker, timespan='day', window=14):
    """
    Calculate RSI using Polygon's built-in RSI endpoint
    
    Args:
        ticker (str): Stock ticker symbol
        timespan (str): Time interval ('day' by default)
        window (int): RSI period (14 by default)
    
    Returns:
        float: RSI value or np.nan if calculation fails
    """
    try:
        url = f"https://api.polygon.io/v1/indicators/rsi/{ticker}"
        params = {
            'timespan': timespan,
            'adjusted': True,
            'window': window,
            'series_type': 'close',
            'order': 'desc',
            'apiKey': API_KEY
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if ('results' in data and 
                'values' in data['results'] and 
                len(data['results']['values']) > 0):
                
                rsi_value = data['results']['values'][0]['value']
                logger.info(f"Successfully fetched RSI for {ticker}: {rsi_value:.2f}")
                return rsi_value
            else:
                logger.warning(f"No RSI data available for {ticker}")
                return np.nan
        else:
            logger.error(f"Error fetching RSI for {ticker}: {response.status_code}")
            return np.nan
            
    except Exception as e:
        logger.error(f"Exception calculating RSI for {ticker}: {str(e)}")
        return np.nan

# Calculate the percentage gap between closing price and SMA
def calculate_gap(closing_price, sma):
    if sma is None:
        return None  # If there's no SMA, gap can't be calculated
    return ((closing_price - sma) / sma) * 100


# Save results to CSV
def save_to_csv(output_csv, results):
    df = pd.DataFrame(results)
    
    # Define the exact column order we want
    columns = [
        'Ticker',
        'Closing Price',
        '200 SMA',
        'Gap',
        'RSI',
        'MACD',
        'MACD Signal',
        'Analysis'
    ]
    
    # Rename columns if needed
    rename_map = {
        'Price': 'Closing Price',
        'SMA': '200 SMA',
        'Signal': 'MACD Signal'
    }
    df = df.rename(columns=rename_map)
    
    # Reorder columns and handle any missing columns
    existing_columns = [col for col in columns if col in df.columns]
    df = df[existing_columns]
    
    # Save to CSV with proper encoding for special characters
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"Results saved to {output_csv}")


def calculate_macd(ticker, timespan='day', short_window=12, long_window=26, signal_window=9):
    """
    Calculate MACD using Polygon's built-in MACD endpoint
    
    Args:
        ticker (str): Stock ticker symbol
        timespan (str): Time interval ('day' by default)
        short_window (int): Short-term EMA period (12 by default)
        long_window (int): Long-term EMA period (26 by default)
        signal_window (int): Signal line period (9 by default)
    
    Returns:
        dict: MACD values or None if calculation fails
            - 'macd': MACD line value
            - 'signal': Signal line value
            - 'histogram': MACD histogram value
    """
    try:
        url = f"https://api.polygon.io/v1/indicators/macd/{ticker}"
        params = {
            'timespan': timespan,
            'adjusted': True,
            'short_window': short_window,
            'long_window': long_window,
            'signal_window': signal_window,
            'series_type': 'close',
            'apiKey': API_KEY
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if ('results' in data and 
                'values' in data['results'] and 
                len(data['results']['values']) > 0):
                
                latest = data['results']['values'][0]
                macd_data = {
                    'macd': latest['value'],
                    'signal': latest['signal'],
                    'histogram': latest['histogram']
                }
                logger.info(f"Successfully fetched MACD for {ticker}: {macd_data}")
                return macd_data
            else:
                logger.warning(f"No MACD data available for {ticker}")
                return None
        else:
            logger.error(f"Error fetching MACD for {ticker}: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Exception calculating MACD for {ticker}: {str(e)}")
        return None


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

def get_latest_price(ticker):
    """
    Get the latest closing price for a ticker using Polygon's API
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        float: Latest closing price or None if fetch fails
    """
    try:
        url = f"{BASE_URL_AGG}{ticker}/prev?apiKey={API_KEY}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if ('results' in data and 
                len(data['results']) > 0 and 
                'c' in data['results'][0]):
                
                closing_price = data['results'][0]['c']
                logger.info(f"Successfully fetched latest price for {ticker}: {closing_price}")
                return closing_price
            else:
                logger.warning(f"No price data available for {ticker}")
                return None
        else:
            logger.error(f"Error fetching price for {ticker}: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Exception getting latest price for {ticker}: {str(e)}")
        return None


def main(input_csv, output_csv):
    tickers = load_tickers(input_csv)
    results = []

    for ticker in tickers:
        logger.info(f"Processing ticker: {ticker}")
        try:
            closing_price = get_latest_price(ticker)
            sma = get_200_day_sma(ticker)
            gap = calculate_gap(closing_price, sma)
            rsi = calculate_rsi(ticker)
            macd_data = calculate_macd(ticker)
            
            analysis = determine_action(gap, rsi, macd_data)
            
            result = {
                'Ticker': ticker,
                'Closing Price': closing_price,
                '200 SMA': sma,
                'Gap': gap,
                'RSI': rsi,
                'MACD': macd_data['macd'] if macd_data else None,
                'MACD Signal': macd_data['signal'] if macd_data else None,
                'Analysis': analysis
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            results.append({
                'Ticker': ticker, 
                'Closing Price': None,
                '200 SMA': None,
                'Gap': None,
                'RSI': None,
                'MACD': None,
                'MACD Signal': None,
                'Analysis': f'Error in Processing: {str(e)}'
            })
        

    save_to_csv(output_csv, results)
    logger.info(f"Results saved to {output_csv}")

# Run the program
if __name__ == "__main__":
    logger.info("Starting the program")
    main('input_tickers.csv', 'output_data.csv')
    logger.info("Program completed")