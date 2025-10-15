import math
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


def download_data(tickers: List[str], start: str = "2023-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Download Adjusted Close prices for tickers between start and end (inclusive).
    Returns DataFrame indexed by business days (Date) and columns = tickers.
    """
    if len(tickers) == 0:
        raise ValueError("Tickers list is empty.")

    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    if 'Adj Close' in df.columns.get_level_values(0):
        adj_df = df['Adj Close']

    idx = pd.bdate_range(start=start, end=end)
    adj_df.reindex(idx)
    adj_df.index.name = 'Date'

    return adj_df


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with any NaN value
    """
    filtered_df = df.dropna(axis=0)
    return filtered_df



def save_clean_df(df: pd.DataFrame, filename: str = 'cleaned_data.xlsx') -> None:
    """
    Save cleaned DataFrame to an xlsx file.
    """
    df.to_excel(f'{filename}.xlsx', index=True)
    df.to_csv(f'{filename}.csv')


if __name__ == '__main__':
    
    tickers = [ "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ",
                "PG", "MA", "DIS", "HD", "BAC", "XOM", "KO", "PFE", "CSCO", "ADBE",
                "AVGO", "ORCL", "WMT", "TSM", "LLY", "NFLX", "XOM", "PLTR", "COST", "ASML",
                "HSBC", "CRM", "TM", "CAT", "MS", "NVS", "IBM", "GE", "UNH", "SAP", 
                "AMD", "CVX", "KO", "BABA", "HD", "WFC", "PM", "GS", "C", "AMGN",
                "QCOM", "AMAT", "LRCX", "NEE", "BKNG", "VZ", "BA", "APH", "SAN", "ETN"
                ]

    adj_df = download_data(tickers, start="2023-01-01", end="2024-12-31")
    print("Data downloaded")
    adj_df = filter_df(adj_df)
    print("Data cleaned")
    save_clean_df(adj_df, filename = 'cleaned_data')
    print("Data saved")