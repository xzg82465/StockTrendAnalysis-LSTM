# Stock Prediction with Transfer Learning

This project implements a **Two-Stage Transfer Learning** approach to predict stock prices using Deep Neural Networks (DNN) with TensorFlow and Keras.

## Concept

Instead of training a model from scratch for a single stock (which often lacks sufficient data for deep learning), this project leverages knowledge transfer:

1.  **General Pre-training:** A model is trained on *all* available stocks (S&P 500 + Taiwan 50) to learn general market dynamics, technical patterns, and sector trends.
2.  **Particular Fine-tuning:** The general model is cloned and fine-tuned on the specific target stock (e.g., TSMC/2330.TW). This specializes the model for the specific asset.

## Pipeline

The workflow is managed by the `StockDataManager` class:

1.  **Data Acquisition**:
    *   Scrapes S&P 500 and Taiwan 50 ticker lists from Wikipedia.
    *   Downloads historical OHLCV data using `yfinance`.
2.  **Feature Engineering**:
    *   Calculates technical indicators: Returns, Log Returns, Moving Averages (MA5, MA20), Volatility, ROC, and RSI.
3.  **Training (Transfer Learning)**:
    *   **General Model**: Trained on the entire dataset.
    *   **Particular Model**: Inherits weights from General Model, trained on target ticker.
4.  **Prediction**:
    *   Loads the specialized Particular Model.
    *   Fetches the most recent data.
    *   Predicts the next day's closing price.

## Environment Setup

Before running the project, you need to set up a Python environment. You can choose between a pure Python setup or using Anaconda.

### 1. Python Environment Setup

#### Option A: Pure Python (Lightweight)

**Windows:**
1.  Download Python from python.org.
2.  Run the installer and **check "Add Python to PATH"**.
3.  Open Command Prompt (cmd) or PowerShell.
4.  Navigate to the project folder.
5.  Create a virtual environment:
    ```cmd
    python -m venv venv
    ```
6.  Activate it:
    ```cmd
    .\venv\Scripts\activate
    ```

**Mac:**
1.  Install via Homebrew (recommended): `brew install python`
2.  Open Terminal and navigate to the project folder.
3.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    ```
4.  Activate it:
    ```bash
    source venv/bin/activate
    ```

**Ubuntu:**
1.  Install Python and venv:
    ```bash
    sudo apt update
    sudo apt install python3 python3-venv python3-pip
    ```
2.  Navigate to the project folder.
3.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    ```
4.  Activate it:
    ```bash
    source venv/bin/activate
    ```

#### Option B: Anaconda / Miniconda (Recommended for Data Science)

1.  Download and install Anaconda or Miniconda.
2.  Open Terminal (Mac/Linux) or Anaconda Prompt (Windows).
3.  Create a new environment:
    ```bash
    conda create -n stock_env python=3.10
    ```
4.  Activate the environment:
    ```bash
    conda activate stock_env
    ```

### 2. VS Code Setup

1.  **Install VS Code:** Download from code.visualstudio.com.
2.  **Install Extensions:**
    Open VS Code, go to the Extensions view (Ctrl+Shift+X), and install the following suggested extensions:
    *   **Python** (Microsoft): Essential for Python support (IntelliSense, linting, debugging).
    *   **Pylance** (Microsoft): Provides performant language support.
    *   **Jupyter** (Microsoft): Useful if you want to explore data using Notebooks.
    *   **Black Formatter** (Microsoft): For automatic code formatting.
3.  **Select Interpreter:**
    *   Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on Mac).
    *   Type `Python: Select Interpreter`.
    *   Select the environment you created earlier (e.g., `venv` or `stock_env`).

## Installation

Once your environment is set up and activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

*Note: If you have a GPU, ensure you have the appropriate CUDA libraries installed, or the code will fallback to CPU.*

## Usage

The entry point for the project is `src/main.py`. You can control the workflow by modifying the parameters inside the `practice_transfer_learning` function.

### 1. Modifying Parameters

Open `src/main.py` and locate `practice_transfer_learning()`:

```python
def practice_transfer_learning():
    # ... imports ...
    
    # --- Control Flags ---
    isUpdateModel = False  # Set to True to download fresh data from Yahoo Finance
    isTrainModel = True    # Set to True to retrain the models (General and Particular)
    isPredict = False      # Set to True to run a prediction

    stock_manager = StockDataManager()

    # --- Data Fetching ---
    if isUpdateModel:
        # Fetches tickers and downloads data up to the specified end_date
        stock_manager.fillTw50Tickers() 
        stock_manager.fillSp500Tickers()
        stock_manager.fetch_data(end_date="2026-03-08")
    else:
        print("Populating ticker from files...")
        # Loads existing raw CSV data from local disk without reprocessing
        stock_manager.load_raw_data(isProcessData=False)

    # --- Training Configuration ---
    if isTrainModel:
        # ticker: The target stock symbol (e.g., "2330.TW" for TSMC, "AAPL" for Apple)
        stock_manager.trainModel(
            ticker="2330.TW", 
            epochs_general=50, 
            epochs_particular=50
        )

    # --- Prediction Configuration ---
    if isPredict:
        # Predicts the price for the specific date based on previous data
        stock_manager.predict_day(ticker="2330.TW", date="2026-03-09")
```

### 2. Running the Project

Run the script from your terminal:

```bash
python src/main.py
```