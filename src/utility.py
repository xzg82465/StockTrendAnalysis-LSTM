import datetime
import yfinance as yf
import pandas as pd
import logging
from yfinance.search import Search
import numpy as np

class SP500Sector:
    MATERIALS = "Materials"
    COMMUNICATION_SERVICES = "Communication Services"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    ENERGY = "Energy"
    FINANCIALS = "Financials"
    HEALTH_CARE = "Health Care"
    INDUSTRIALS = "Industrials"
    OTHER = "Taiwan Other"
    REAL_ESTATE = "Real Estate"
    INFORMATION_TECHNOLOGY = "Information Technology"
    UTILITIES = "Utilities"

    @classmethod
    def get_all_sectors(cls):
        return sorted([v for k, v in vars(cls).items() if k.isupper() and isinstance(v, str)])

    @classmethod
    def get_sector_id(cls, sector_str):
        sectors = cls.get_all_sectors()
        if sector_str in sectors:
            return sectors.index(sector_str)
        return sectors.index(cls.OTHER)

    def isMember(self, sector_str):
        return sector_str in self.get_all_sectors()

    def fromYfinanceSector(self, sector_str):
        if sector_str == "Basic Materials":
            return self.MATERIALS
        elif sector_str == "Communication Services":
            return self.COMMUNICATION_SERVICES
        elif sector_str == "Consumer Cyclical":
            return self.CONSUMER_DISCRETIONARY
        elif sector_str == "Consumer Defensive":
            return self.CONSUMER_STAPLES
        elif sector_str == "Energy":
            return self.ENERGY
        elif sector_str == "Financial Services":
            return self.FINANCIALS
        elif sector_str == "Healthcare":
            return self.HEALTH_CARE
        elif sector_str == "Industrials":
            return self.INDUSTRIALS
        elif sector_str == "Real Estate":
            return self.REAL_ESTATE
        elif sector_str == "Technology":
            return self.INFORMATION_TECHNOLOGY
        elif sector_str == "Utilities":
            return self.UTILITIES
        else:
            return self.OTHER

def get_yfinance_ticker_info(code) -> dict:
    """
    Get ticker information using yfinance search method, which is faster than downloading history.
    Returns a dictionary with keys
    """
    if not hasattr(get_yfinance_ticker_info, "call_counter"):
        get_yfinance_ticker_info.call_counter = 0

    get_yfinance_ticker_info.call_counter += 1
    # Suppress yfinance logging to avoid cluttering output
    logger = logging.getLogger('yfinance')
    original_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        call_count = get_yfinance_ticker_info.call_counter
        results = Search(code, max_results=10).quotes
        if call_count == 1:
            print(results[0])
            
        print(f"{get_yfinance_ticker_info.call_counter}. Searched results for '{code}': {[x.get('symbol') for x in results]}")
        for r in results:
            if r.get("symbol") == code:
                return {k: r.get(k, '') for k in ["exchange", "symbol", "longname", "sector"]}
        
        for r in results:
            if r.get("symbol", "").startswith(code):
                return {k: r.get(k, '') for k in ["exchange", "symbol", "longname", "sector"]}
        
        if results:
            r = results[0]
            return {k: r.get(k, '') for k in ["exchange", "symbol", "longname", "sector"]}
        
        return None
            
    except Exception as e:
        print(f"Warning: Search method failed for '{code}' with error: {e}.")
    finally:
        logger.setLevel(original_level)

    print(f"Warning: Could not find valid ticker for {code} using search.")
    return None

def get_ticker_start_date(code):
    """
    Get the start date (first trade date) of a stock without downloading all data.
    """
    try:
        ticker = yf.Ticker(code)
        # get_history_metadata fetches lightweight metadata about the listing
        meta = ticker.get_history_metadata()
        return meta.get("firstTradeDate", None)
    except Exception as e:
        print(f"Warning: Could not fetch start date for '{code}': {e}")
    return None

def calculate_return(df: pd.DataFrame) -> pd.Series:
    return df['Close'].pct_change()

def calculate_log_return(df: pd.DataFrame) -> pd.Series:
    return np.log(df['Close'] / df['Close'].shift(1))

def calculate_ma_5(df: pd.DataFrame) -> pd.Series:
    return df['Close'].rolling(window=5).mean()

def calculate_ma_20(df: pd.DataFrame) -> pd.Series:
    return df['Close'].rolling(window=20).mean()

def calculate_vol_20(df: pd.DataFrame) -> pd.Series:
    return df['Return'].rolling(window=20).std()

def calculate_roc_10(df: pd.DataFrame) -> pd.Series:
    return (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)

def calculate_rsi_14(df: pd.DataFrame) -> pd.Series:
    window = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df: pd.DataFrame) -> pd.Series:
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    return ema_12 - ema_26

def calculate_macd_signal(df: pd.DataFrame) -> pd.Series:
    macd = calculate_macd(df)
    return macd.ewm(span=9, adjust=False).mean()

def calculate_kd_k(df: pd.DataFrame, n: int = 9) -> pd.Series:
    if 'High' not in df.columns or 'Low' not in df.columns:
        return pd.Series(50.0, index=df.index)
    low_min = df['Low'].rolling(window=n).min()
    high_max = df['High'].rolling(window=n).max()
    rsv = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-8)
    # com=2 is exponentially equivalent to alpha=1/3, matching standard KD smoothing
    return rsv.ewm(com=2, adjust=False).mean()

def calculate_kd_d(df: pd.DataFrame) -> pd.Series:
    k = calculate_kd_k(df)
    return k.ewm(com=2, adjust=False).mean()

def calculate_bb_upper(df: pd.DataFrame) -> pd.Series:
    ma_20 = calculate_ma_20(df)
    std_20 = df['Close'].rolling(window=20).std()
    return ma_20 + 2 * std_20

def calculate_bb_lower(df: pd.DataFrame) -> pd.Series:
    ma_20 = calculate_ma_20(df)
    std_20 = df['Close'].rolling(window=20).std()
    return ma_20 - 2 * std_20

def calculate_bias_6(df: pd.DataFrame) -> pd.Series:
    ma_6 = df['Close'].rolling(window=6).mean()
    return (df['Close'] - ma_6) / ma_6 * 100

feature_map = {
    'Return': calculate_return,
    'Log_Return': calculate_log_return,
    'MA_5': calculate_ma_5,
    'MA_20': calculate_ma_20,
    'Vol_20': calculate_vol_20,
    'ROC_10': calculate_roc_10,
    'RSI_14': calculate_rsi_14,
    'MACD': calculate_macd,
    'MACD_Signal': calculate_macd_signal,
    'KD_K': calculate_kd_k,
    'KD_D': calculate_kd_d,
    'BB_Upper': calculate_bb_upper,
    'BB_Lower': calculate_bb_lower,
    'BIAS_6': calculate_bias_6
}

def generate_features(df: pd.DataFrame, sector_str: str = SP500Sector.OTHER, is_training: bool = True) -> pd.DataFrame:
    for feature_name, calc_func in feature_map.items():
        df[feature_name] = calc_func(df)
        
    df['Sector_ID'] = SP500Sector.get_sector_id(sector_str)
    
    if is_training:
        df['Target'] = df['Close'].shift(-1) / df['Close'] - 1.0
        
    return df
