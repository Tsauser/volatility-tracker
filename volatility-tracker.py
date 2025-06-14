import yfinance as yf
import pandas as pd
import sqlite3
import smtplib
from datetime import datetime, timedelta, date
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()  # Loads .env variables into environment

TICKERS = [
    ticker.strip().upper() for ticker in os.getenv("TICKERS", "").split(",")
    if ticker.strip()
]
EMAIL_FROM = os.getenv("EMAIL_FROM", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_TO = os.getenv("EMAIL_TO", "")

RSI_OVERSOLD_THRESHOLD = float(os.getenv("RSI_OVERSOLD_THRESHOLD", 40))
IV_LOW_PERCENTILE = float(os.getenv("IV_LOW_PERCENTILE", 0.20))
RSI_WEIGHT = float(os.getenv("RSI_WEIGHT", 0.5))

DB_FILE = "leaps_iv_rsi_data.db"

# --- DATABASE SETUP ---
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS iv_rsi_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        date TEXT,
        expiry TEXT,
        strike REAL,
        iv REAL,
        rsi REAL
    )
''')
conn.commit()


def calculate_iv_percentile(current_iv, all_iv_values):
    return np.sum(all_iv_values < current_iv) / len(all_iv_values)


# --- EMAIL CLIENT ---
def send_email(message, subject="LEAPS Call IV Alert"):
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_FROM, EMAIL_PASSWORD)
        smtp.send_message(msg)


# --- IV & RSI FETCHING ---
def get_leaps_calls(ticker_obj):
    today = datetime.today()
    leaps_cutoff = today + timedelta(days=270)  # ~9 months ahead
    leaps_expiries = [
        d for d in ticker_obj.options
        if datetime.strptime(d, '%Y-%m-%d') > leaps_cutoff
    ]

    if not leaps_expiries:
        return None

    # Use the furthest expiry date (last in list)
    leaps_expiry = leaps_expiries[-1]
    options = ticker_obj.option_chain(leaps_expiry)
    return options.calls, leaps_expiry


def calculate_rsi(ticker_obj, period=14):
    # Use close prices to calculate RSI
    hist = ticker_obj.history(period='1mo', interval='1d')
    delta = hist['Close'].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    if not rsi.empty:
        return rsi.iloc[-1]
    else:
        return None


def save_iv_rsi(ticker, expiry, strike, iv, rsi):
    cursor.execute(
        '''
        INSERT INTO iv_rsi_history (ticker, date, expiry, strike, iv, rsi)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (ticker, date.today().isoformat(), expiry, strike, iv, rsi))
    conn.commit()


def get_iv_threshold(ticker):
    print(f"Getting IV threshold for {ticker,}...")
    df = pd.read_sql_query('SELECT iv FROM iv_rsi_history WHERE ticker = ?',
                           conn,
                           params=(ticker))
    if df.empty:
        return None
    return df['iv'].quantile(IV_LOW_PERCENTILE)


def get_average_rsi(ticker):
    print(f"Getting average RSI for {ticker}...")
    df = pd.read_sql_query('SELECT rsi FROM iv_rsi_history WHERE ticker = ?',
                           conn,
                           params=(ticker, ))
    if df.empty:
        return None
    return df['rsi'].mean()


def get_all_historical_ivs(ticker):
    print(f"Getting historical IVs for {ticker}...")
    df = pd.read_sql_query('SELECT iv FROM iv_rsi_history WHERE ticker = ?',
                           conn,
                           params=(ticker, ))
    return df['iv'].values if not df.empty else []


# --- Main Processing ---
def process_ticker(ticker_symbol):
    print(f"Checking {ticker_symbol}...\n")

    ticker_obj = yf.Ticker(ticker_symbol)

    # Get RSI
    rsi = calculate_rsi(ticker_obj)
    if rsi is None:
        print("RSI unavailable\n")
        return

    # Get LEAPS calls
    leaps_calls, expiry = get_leaps_calls(ticker_obj)
    if leaps_calls is None or leaps_calls.empty:
        print("No LEAPS calls available\n")
        return

    # Filter to ATM strike (closest to current price)
    current_price = ticker_obj.history(period='1d')['Close'].iloc[-1]
    print(f"GME last recorded price: {round(current_price, 2)}\n")

    leaps_calls['strike_diff'] = abs(leaps_calls['strike'] - current_price)
    atm_call = leaps_calls.loc[leaps_calls['strike_diff'].idxmin()]

    current_iv = atm_call['impliedVolatility']
    strike = atm_call['strike']

    # Save current data
    save_iv_rsi(ticker_symbol, expiry, strike, current_iv, rsi)

    # Get historical IVs and compute percentile rank
    historical_ivs = get_all_historical_ivs(ticker_symbol)
    if len(historical_ivs) < 10:
        print("Not enough IV history yet to determine percentile.\n")
        return

    iv_percentile = calculate_iv_percentile(current_iv, historical_ivs)
    iv_score = 1 - iv_percentile  # Lower percentile => higher score

    # Normalize RSI to 0-1 (lower RSI = more oversold = higher score)
    rsi_score = max(0, (RSI_OVERSOLD_THRESHOLD - rsi) / RSI_OVERSOLD_THRESHOLD)

    combined_score = (iv_score * (1 - RSI_WEIGHT)) + (rsi_score * RSI_WEIGHT)

    print(
        f"Current IV: {current_iv:.2%}, IV Percentile: {iv_percentile:.2%}, RSI: {rsi:.2f}"
    )
    print(
        f"Scores => IV: {iv_score:.2f}, RSI: {rsi_score:.2f}, Combined: {combined_score:.2f}\n"
    )
    print(
        f"Current IV: {current_iv:.2%}, Threshold IV: {iv_threshold:.2%}, RSI: {rsi:.2f}\n"
    )
    print(
        f"Scores => IV: {iv_score:.2f}, RSI: {rsi_score:.2f}, Combined: {combined_score:.2f}\n"
    )

    # Alert if combined score high enough (e.g., > 0.7)
    if iv_percentile <= IV_LOW_PERCENTILE and combined_score > 0.7:
        message = (f"{ticker_symbol} LEAPS Call IV Alert!\n"
                   f"Date: {date.today().isoformat()}\n"
                   f"Expiry: {expiry}\n"
                   f"Strike: {strike}\n"
                   f"IV: {current_iv:.2%} (Percentile: {iv_percentile:.2%})\n"
                   f"RSI: {rsi:.2f}\n"
                   f"Score: {combined_score:.2f}\n"
                   f"Consider buying calls.")
        send_email(message, subject=f"{ticker_symbol} LEAPS Call IV Alert")
        print("Email alert sent!\n")
    else:
        print("No alert triggered.\n")


if __name__ == "__main__":

    for ticker in TICKERS:
        try:
            process_ticker(ticker)
        except Exception as e:
            print(f"Error processing {ticker}: {e}\n")

    conn.close()
