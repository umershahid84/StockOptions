import pandas as pd
from sklearn.ensemble import IsolationForest
import yfinance as yf
from tabulate import tabulate
import time
from colorama import Fore, Style, init
from datetime import datetime
import pytz
import numpy as np

init(autoreset=True)
est = pytz.timezone('US/Eastern')
major_companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V', 'WMT', 'PG', 'NVDA', 'DIS', 'MA', 'HD', 'UNH', 'VZ', 'PYPL', 'ADBE', 'NFLX', 'SPY']

def fetch_options_data(ticker):
    stock = yf.Ticker(ticker)
    calls_list, puts_list = [], []
    try:
        for exp in stock.options:
            options = stock.option_chain(exp)
            options.calls['expirationDate'], options.puts['expirationDate'] = exp, exp
            calls_list.append(options.calls)
            puts_list.append(options.puts)
        return pd.concat(calls_list, ignore_index=True), pd.concat(puts_list, ignore_index=True)
    except Exception as e:
        print(f"Error fetching options data for {ticker}: {e}")
        return None, None

def preprocess_data(df):
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

def create_features(df):
    df['volume'] = df['volume'] / 1_000_000
    df['volume_change'] = df['volume'].pct_change()
    df['open_interest_change'] = df['openInterest'].pct_change()
    df['implied_volatility_change'] = df['impliedVolatility'].pct_change()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df[['volume_change', 'implied_volatility_change', 'open_interest_change']], df

def train_model(features):
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(features)
    return model

def detect_anomalies(model, features):
    return model.predict(features)

def highlight_anomalies(row):
    color = Fore.GREEN if row['Type'] == 'Call' else Fore.RED
    return [color + str(value) + Style.RESET_ALL for value in row]

def format_volume(volume):
    return f"{int(volume)} mil" if volume >= 10 else f"{volume:.2f} mil"

def determine_sentiment(call_put_ratio):
    return 'Bullish' if call_put_ratio > 1 else 'Bearish'

def report_anomalies(ticker, call_df, call_anomalies, put_df, put_anomalies):
    call_anomalies_df, put_anomalies_df = call_df[call_anomalies == -1].copy(), put_df[put_anomalies == -1].copy()
    call_anomalies_df['Type'], put_anomalies_df['Type'] = 'Call', 'Put'
    anomalies_df = pd.concat([call_anomalies_df, put_anomalies_df])
    anomalies_df['Call/Put Ratio'] = anomalies_df['volume'] / anomalies_df['openInterest']
    anomalies_df['Average Entry Price'] = (anomalies_df['volume'] * anomalies_df['lastPrice']).cumsum() / anomalies_df['volume'].cumsum()
    anomalies_df['lastTradeDate'] = pd.to_datetime(anomalies_df['lastTradeDate'])
    anomalies_df['lastTradeDate_EST'] = anomalies_df['lastTradeDate'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern') if anomalies_df['lastTradeDate'].dt.tz is None else anomalies_df['lastTradeDate'].dt.tz_convert('US/Eastern')
    anomalies_df['lastTradeDate_EST'] = anomalies_df['lastTradeDate_EST'].dt.strftime('%Y-%m-%d %H:%M:%S')
    anomalies_df['Sentiment'] = anomalies_df['Call/Put Ratio'].apply(determine_sentiment)
    anomalies_df.drop(columns=['lastTradeDate'], inplace=True)
    anomalies_df.insert(1, 'lastTradeDate_EST', anomalies_df.pop('lastTradeDate_EST'))
    anomalies_df.insert(anomalies_df.columns.get_loc('ask') + 1, 'Average Entry Price', anomalies_df.pop('Average Entry Price'))
    top_anomalies_df = anomalies_df.nlargest(5, 'volume')
    if not top_anomalies_df.empty:
        top_anomalies_df['volume'] = top_anomalies_df['volume'].apply(format_volume)
        print(f"Top 5 Anomalies for {ticker}:")
        highlighted_anomalies = top_anomalies_df.apply(highlight_anomalies, axis=1)
        print(tabulate(highlighted_anomalies.values.tolist(), headers=top_anomalies_df.columns, tablefmt='grid'))
        print("\n")

def track_unusual_options_whales(companies):
    while True:
        for ticker in companies:
            try:
                calls, puts = fetch_options_data(ticker)
                if calls is None or puts is None or calls.empty or puts.empty:
                    continue
                calls, puts = preprocess_data(calls), preprocess_data(puts)
                call_features, call_df = create_features(calls)
                put_features, put_df = create_features(puts)
                if call_features.empty or put_features.empty:
                    continue
                call_model, put_model = train_model(call_features), train_model(put_features)
                call_anomalies, put_anomalies = detect_anomalies(call_model, call_features), detect_anomalies(put_model, put_features)
                report_anomalies(ticker, call_df, call_anomalies, put_df, put_anomalies)
                time.sleep(1)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
        time.sleep(180)

track_unusual_options_whales(major_companies)
