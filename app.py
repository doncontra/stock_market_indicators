# from polygon import RESTClient

# client = RESTClient(api_key="ppisGP51Q3ZPNYFSX8hCdWleFNzKa5I0")

# ticker = "JMIA"

import pandas as pd
import requests
import time

# Constants
API_KEY = 'ppisGP51Q3ZPNYFSX8hCdWleFNzKa5I0'
BASE_URL_AGG = 'https://api.polygon.io/v2/aggs/ticker/'  # For closing prices
BASE_URL_SMA = 'https://api.polygon.io/v1/indicators/sma/'  # For SMA
CALL_LIMIT = 2  # API calls per minute
DELAY = 60 / CALL_LIMIT  # Delay to respect API rate limit


# Read the tickers from a CSV file
def load_tickers(input_csv):
    df = pd.read_csv(input_csv)
    return df['Ticker'].tolist()

# Fetch closing prices for a ticker from Polygon API
def get_closing_prices(ticker):
    url = f"{BASE_URL_AGG}{ticker}/prev?apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print(ticker, " OK")
        if 'results' in data:
            return [result['c'] for result in data['results']]  # Extract closing prices
    else:
        print(f"Error fetching data for {ticker}: {response.status_code}")
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

# Calculate the percentage gap between closing price and SMA
def calculate_gap(closing_price, sma):
    if sma is None:
        return None  # If there's no SMA, gap can't be calculated
    return ((closing_price - sma) / sma) * 100

# Save results to CSV
def save_to_csv(output_csv, results):
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

def main(input_csv, output_csv):
    tickers = load_tickers(input_csv)
    results = []

    for ticker in tickers:
        closing_prices = get_closing_prices(ticker)  # Fetch closing prices
        if closing_prices: 
            closing_price = closing_prices[-1]  # Get the last closing price
            sma = get_200_day_sma(ticker)  # Fetch the 200-day SMA
            gap = calculate_gap(closing_price, sma)  # Calculate the gap
            results.append({'Ticker': ticker, 'Closing Price': closing_price, '200 SMA': sma, 'Gap': gap})
        
        # Respect rate limit
        time.sleep(DELAY)

    save_to_csv(output_csv, results)

# Run the program
main('input_tickers.csv', 'output_data.csv')
