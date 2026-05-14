import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
from sklearn import metrics
import datetime
from sklearn.neural_network import MLPClassifier


def test_StockManager():
    from stock_manager import StockDataManager
    stock_manager = StockDataManager()
    stock_manager.fillTw50Tickers()
    print(stock_manager)
    # stock_manager.fillSp500Tickers()
    # print(stock_manager)
    # for stock in stock_manager.tw50_tickers:
    #     stock_manager.fillRelatedTickers(stock)
    
    stock_manager.fetch_data(end_date="2026-03-31")
    return

def practice_transfer_learning():
    # Import the new class
    from stock_manager import StockDataManager
    
    isUpdateModel = True
    isTrainModel = True
    isPredict = True
    # Initialize Manager
    stock_manager = StockDataManager()

    if isUpdateModel:
        stock_manager.fillTw50Tickers() 
        print(stock_manager)
        stock_manager.fillSp500Tickers()
        print(stock_manager)
        stock_manager.fetch_data(end_date="2026-04-29")
    else:
        print("Populating ticker from files...")
        stock_manager.load_raw_data(isProcessData=False)   # reads CSV files saved by fetch_data()

    if isTrainModel:
        stock_manager.trainModel(ticker="2330.TW", epochs_general=50, epochs_particular=50)

    if isPredict:
        stock_manager.predict_day(ticker="2330.TW", date="2026-04-30")

    return


def practice_mta_model():
    """
    Demonstrates how to train and predict with the MTA model.

    The MTA model is a drop-in replacement for TransferLearningModel.
    It follows the exact same transfer-learning pipeline already
    built into StockDataManager — you just call the mta_* variants.

    Supported loss functions (loss_fn parameter):
        'ce'          - Cross-Entropy          (default, safest)
        'focal'       - Focal Loss             (handles class imbalance)
        'madl_focal'  - MADL-Focal             (thesis novel loss, risk-aware)
    """
    from stock_manager import StockDataManager

    # ── Configuration ────────────────────────────────────────────────────────
    TICKER     = "2330.TW"
    LOSS_FN    = "ce"        # change to 'focal' or 'madl_focal' to use thesis losses
    UPDATE_DATA = False        # set True to re-download from yfinance
    TRAIN       = True
    PREDICT     = False

    stock_manager = StockDataManager()

    # ── 1. Load (or re-download) price data ──────────────────────────────────
    if UPDATE_DATA:
        stock_manager.fillTw50Tickers()
        # stock_manager.fillSp500Tickers()
        stock_manager.fetch_data(end_date="2026-03-08")
    else:
        stock_manager.load_raw_data(isProcessData=False)   # reads CSV files saved by fetch_data()

    # ── 2. Three-stage MTA training ──────────────────────────────────────────
    if TRAIN:
        stock_manager.trainMTAModel(
            ticker=TICKER,
            epochs_general=50,      # General pre-training  (all stocks)
            epochs_particular=50,   # Stock fine-tuning     (TICKER only)
            loss_fn=LOSS_FN,
        )

    # ── 3. Predict next-day direction ─────────────────────────────────────────
    if PREDICT:
        signal = stock_manager.predict_day_mta(ticker=TICKER, date="2026-03-09")
        # signal > 0  →  bullish  (bull prob > bear prob)
        # signal < 0  →  bearish
        # signal ~ 0  →  sideways / uncertain
        print(f"\nFinal signal for {TICKER}: {signal:+.4f}")

    return

if __name__ == "__main__":
    # test_StockManager()
    practice_transfer_learning()
    # p0ractice_mta_model()             # NEW: MTA model (thesis architecture)
