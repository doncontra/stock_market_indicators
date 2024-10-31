import pandas as pd
import requests
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Constants
BASE_URL_AGG = 'https://api.polygon.io/v2/aggs/ticker/'
BASE_URL_SMA = 'https://api.polygon.io/v1/indicators/sma/'

# Set page config at the very beginning
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# Initialize session state for API key
if 'API_KEY' not in st.session_state:
    st.session_state.api_key = ''

# API key input
st.sidebar.title("Configuration")
API_KEY = st.sidebar.text_input("Enter your Paid Polygon API key:", value=st.session_state.api_key, type="password")
if API_KEY:
    st.session_state.api_key = API_KEY


# Set up logger
def setup_logger():
    logger = logging.getLogger('polygon_api_logger')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = RotatingFileHandler('polygon_api.log', maxBytes=1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()

# Functions from your original script
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

def calculate_gap(closing_price, sma):
    if sma is None:
        return None  # If there's no SMA, gap can't be calculated
    return ((closing_price - sma) / sma) * 100

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

    # Sweet Spot Check
    if (gap > 0 and  # Above 200 SMA
        30 <= rsi <= 70 and  # RSI between 30-70
        macd_data['macd'] > macd_data['signal'] and  # Bullish crossover
        macd_data['histogram'] > 0):  # Rising histogram
        
        action = "Sweet Spot"
        reasons = [
            f"Primary: Trading above 200 SMA (Gap: {gap:.1f}%)",
            f"Supporting: RSI in optimal range ({rsi:.1f})",
            f"Confirming: MACD shows bullish momentum (Crossover: {macd_data['macd']:.3f} > {macd_data['signal']:.3f})",
            f"Confirming: Rising MACD histogram ({macd_data['histogram']:.3f})"
        ]
        
        # Additional strength indicators for near-oversold condition
        if 30 <= rsi <= 40:
            reasons.append("Bonus: RSI near oversold territory - potential reversal point")
    
# Regular Action Determination (only if not Sweet Spot)
    elif gap > 0:
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

    # Format the analysis as a single string
    if action == "No Action" and not warnings:
        action = "No Warning - No Action"
        
    analysis = f"{action}\n\nReasons:\n"
    analysis += "\n".join(f"- {reason}" for reason in reasons)
    
    if strength:
        analysis += "\n\nStrength Indicators:\n"
        analysis += "\n".join(f"- {indicator}" for indicator in strength)
    
    if warnings:
        analysis += "\n\nWarnings:\n"
        analysis += "\n".join(f"- {warning}" for warning in warnings)

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

def extract_action(analysis):
    # Extract the action from the analysis string
    return analysis.split('\n')[0]


def main():
    st.title("Stock Analysis Dashboard")

    if 'results_df' not in st.session_state:
        st.session_state.results_df = None

    if not st.session_state.api_key:
        st.warning("Please enter your Polygon API key in the sidebar to proceed.")
        return

    uploaded_file = st.file_uploader("Upload CSV with tickers", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        tickers = df['Ticker'].tolist()

        if st.button("Analyze Stocks") or st.session_state.results_df is None:
            results = []
            progress_bar = st.progress(0)
            for i, ticker in enumerate(tickers):
                try:
                    # # Pass the API key to all functions that need it
                    # closing_price = get_latest_price(ticker, st.session_state.api_key)
                    # sma = get_200_day_sma(ticker, st.session_state.api_key)
                    # gap = calculate_gap(closing_price, sma)
                    # rsi = calculate_rsi(ticker, st.session_state.api_key)
                    # macd_data = calculate_macd(ticker, st.session_state.api_key)

                    closing_price = get_latest_price(ticker)
                    sma = get_200_day_sma(ticker)
                    gap = calculate_gap(closing_price, sma)
                    rsi = calculate_rsi(ticker)
                    macd_data = calculate_macd(ticker)
                    
                    analysis = determine_action(gap, rsi, macd_data)
                    action = extract_action(analysis)
                    
                    result = {
                        'Ticker': ticker,
                        'Closing Price': closing_price,
                        '200 SMA': sma,
                        'Gap': gap,
                        'RSI': rsi,
                        'MACD': macd_data['macd'] if macd_data else None,
                        'MACD Signal': macd_data['signal'] if macd_data else None,
                        'Action': action,
                        'Analysis': analysis
                    }
                    results.append(result)
                except Exception as e:
                    st.error(f"Error processing {ticker}: {str(e)}")
                    results.append({
                        'Ticker': ticker, 
                        'Closing Price': None,
                        '200 SMA': None,
                        'Gap': None,
                        'RSI': None,
                        'MACD': None,
                        'MACD Signal': None,
                        'Action': 'Error',
                        'Analysis': f'Error in Processing: {str(e)}'
                    })
                progress_bar.progress((i + 1) / len(tickers))

            st.session_state.results_df = pd.DataFrame(results)
            st.success("Analysis complete!")

        if st.session_state.results_df is not None:
            # Action filter
            all_actions = ['All'] + list(st.session_state.results_df['Action'].unique())
            selected_action = st.selectbox("Filter by Action", all_actions)

            if selected_action != 'All':
                filtered_df = st.session_state.results_df[st.session_state.results_df['Action'] == selected_action]
            else:
                filtered_df = st.session_state.results_df

            # Display results
            st.subheader("Analysis Results")
            st.dataframe(filtered_df)

            # Visualizations
            st.subheader("Visualizations")

            # Gap Analysis
            st.subheader("Gap Analysis")
            fig_gap = go.Figure()
            fig_gap.add_trace(go.Bar(
                x=filtered_df['Ticker'],
                y=filtered_df['Gap'],
                name='Gap',
                marker_color=filtered_df['Gap'].apply(lambda x: 'green' if x is not None and x > 0 else 'red')
            ))
            fig_gap.update_layout(title='Gap Analysis by Ticker', xaxis_title='Ticker', yaxis_title='Gap (%)')
            st.plotly_chart(fig_gap, use_container_width=True)

            # RSI Analysis
            st.subheader("RSI Analysis")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=filtered_df['Ticker'],
                y=filtered_df['RSI'],
                mode='markers',
                name='RSI',
                marker=dict(
                    size=10,
                    color=filtered_df['RSI'],
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Oversold")
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.update_layout(title='RSI Analysis by Ticker', xaxis_title='Ticker', yaxis_title='RSI')
            st.plotly_chart(fig_rsi, use_container_width=True)

            # Action distribution pie chart
            st.subheader("Action Distribution")
            fig_pie = px.pie(filtered_df, names='Action', title='Distribution of Actions')
            st.plotly_chart(fig_pie, use_container_width=True)

            # Scatter plot: RSI vs Gap
            st.subheader("RSI vs Gap")
            fig_scatter = px.scatter(
                filtered_df[filtered_df['Gap'].notnull()],  # Filter out rows with None Gap
                x='RSI', 
                y='Gap', 
                color='Action', 
                hover_data=['Ticker', 'Closing Price', '200 SMA'],
                title='RSI vs Gap'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Download results
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download filtered results as CSV",
                data=csv,
                file_name="stock_analysis_results_filtered.csv",
                mime="text/csv",
            )

    else:
        st.info("Please upload a CSV file with a 'Ticker' column to begin analysis.")

if __name__ == "__main__":
    main()
