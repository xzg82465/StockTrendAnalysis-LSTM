import yfinance as yf
import pandas as pd
import os
import requests
import shutil
from io import StringIO
import urllib.parse
import numpy as np
import json
import concurrent.futures
from tqdm import tqdm

from model import TransferLearningModel
from mta_model import MTAStockModel
from utility import *
from stock import Stock


# --- Dynamic Path Configuration ---
# Get the absolute path of the directory containing this script (src/)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from src/)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_DATA_FOLDER = os.path.join(_PROJECT_ROOT, "project_data")

class StockDataManager:
    def __init__(self, data_folder = _DATA_FOLDER):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            
        self.raw_folder = os.path.join(self.data_folder, "rawStock")
        self.processed_folder = os.path.join(self.data_folder, "processedStock")
        
        self.stocks: dict[str, Stock] = {}
        self.sp500_tickers = []
        self.tw50_tickers = []
        self.df_data = pd.DataFrame()
        self.feature_cols = list(feature_map.keys()) + ["Sector_ID"]
        self.general_model: TransferLearningModel = None
        self.particular_model: TransferLearningModel = None

        # MTA counterparts (one per training stage, same lifecycle as above)
        self.mta_general_model: MTAStockModel = None
        self.mta_particular_model: MTAStockModel  = None
        self.finnhub_api_key: str = ""
        self.load_config()
        return

    def __str__(self):
        return f"StockDataManager(data_folder='{self.data_folder}', total_stocks={len(self.stocks)})"

    def load_config(self, config_file="config.json"):
        """Loads configuration (API keys) from a JSON file in the project root."""
        # Assuming src/stock_manager.py, project root is one level up
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, config_file)

        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self.finnhub_api_key = config.get("finnhub_api_key", "")
                    print(f"Loaded configuration from {config_path}")
            except Exception as e:
                print(f"Failed to load config: {e}")
        return

    def fillSp500Tickers(self):
        """Scrapes S&P 500 list from Wikipedia."""
        print("Fetching S&P 500 tickers from Wikipedia...")
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            r = requests.get(url, headers=headers)
            # Read HTML tables
            dfs = pd.read_html(StringIO(r.text), match="Symbol")
            df = dfs[0]
            # Clean symbols (e.g. BRK.B -> BRK-B for yfinance)
            df["yfinance_Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
            self.sp500_tickers = sorted(df["yfinance_Symbol"].tolist())
            # Update sector map
            for ticker in self.sp500_tickers:
                # Check existence using yfinance search, similar to Taiwan 50 logic
                ticker_info = get_yfinance_ticker_info(ticker)
                if ticker_info:
                    if ticker_info["symbol"] in self.stocks:
                        print(f"Warning: Duplicate ticker '{ticker_info['symbol']}' found. Skipping.")
                        print(f"Existing stock info: {self.stocks[ticker_info['symbol']]}")
                        continue
                    
                    self.stocks[ticker_info["symbol"]] = Stock.from_dict(ticker_info)
        
        except Exception as e:
            print(f"Failed to fetch S&P 500: {e}")

        return

    def fillTw50Tickers(self):
        """Fetches TW50 tickers from Wikipedia."""
        print("Fetching Taiwan 50 tickers from Wikipedia...")
        try:
            url = r"https://zh.wikipedia.org/wiki/%E8%87%BA%E7%81%A350%E6%8C%87%E6%95%B8"
            headers = {
                "User-Agent": r"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            r = requests.get(url, headers=headers)
            r.encoding = 'utf-8'
            dfs = pd.read_html(StringIO(r.text))
            
            # Find the table containing '股票代號' (Stock Code)
            df = next((d for d in dfs if "股票代號" in d.columns), None)
            
            if df is not None:
                # Extract tickers from all columns containing '股票代號'
                # Wikipedia table often splits 50 rows into 2 columns (1-25, 26-50)
                target_cols = [c for c in df.columns if "股票代號" in str(c)]
                tickers = []
                for col in target_cols:
                    tickers.extend(df[col].dropna().astype(str).str.strip().tolist())
                
                tw_tickers = [f"{code}.TW" for t in tickers if (code := ''.join(filter(str.isdigit, t)))]
                tw_tickers.sort()
                self.tw50_tickers = tw_tickers
                for ticker in tw_tickers:
                    ticker_info = get_yfinance_ticker_info(ticker)
                    if ticker_info:
                        if ticker_info["symbol"] in self.stocks:
                            print(f"Warning: Duplicate ticker '{ticker_info['symbol']}' found. Skipping.")
                            continue

                        ticker_info["sector"] = SP500Sector().fromYfinanceSector(ticker_info.get("sector", ""))
                        self.stocks[ticker_info["symbol"]] = Stock.from_dict(ticker_info)
            else:
                print("Error: Could not find constituents table on Wikipedia.")
                
        except Exception as e:
            print(f"Failed to fetch Taiwan 50: {e}")
        
        return

    def fetch_data(self, start_date=None, end_date: str=datetime.datetime.now().strftime("%Y-%m-%d")) -> None:
        """Fetches daily data using yfinance."""
        tickers = list(self.stocks.keys())
        print(f"Downloading data for {len(tickers)} tickers...")
        # group_by='ticker' creates a MultiIndex (Ticker, OHLCV)
        if start_date:
            self.df_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True, threads=True)
        else:
            # If no start_date is provided, fetch the full available history ('max')
            self.df_data = yf.download(tickers, period="max", end=end_date, group_by='ticker', auto_adjust=True, threads=True)
        
        # Convert single stock data to MultiIndex format to maintain consistency
        if len(self.stocks) == 1 and not isinstance(self.df_data.columns, pd.MultiIndex):
            ticker = list(self.stocks.keys())[0]
            self.df_data.columns = pd.MultiIndex.from_product([[ticker], self.df_data.columns])
        
        
        shutil.rmtree(self.raw_folder, ignore_errors=True)
        shutil.rmtree(self.processed_folder, ignore_errors=True)
        
        self.save_raw_data()
        self.preProcessing()
        return

    def save_raw_data(self):
        """Saves the raw downloaded data to individual CSV files."""
        print(f"Saving raw data to {self.raw_folder}...")
        
        if self.stocks:
            # Save stock metadata to a CSV for easy loading later, including filename mapping for raw data
            stock_list = []
            for stock in self.stocks.values():
                stock.filename = stock.ticker.replace("/", "_")
                stock_list.append(stock.to_dict())
                
            pd.DataFrame(stock_list).to_csv(os.path.join(self.data_folder, "stock_list.csv"), index=False)

        os.makedirs(self.raw_folder, exist_ok=True)
        
        # Clear existing raw data files
        for f in os.listdir(self.raw_folder):
            if f.endswith(".csv"):
                os.remove(os.path.join(self.raw_folder, f))
                
        tickers = self.df_data.columns.levels[0]
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._save_single_raw_ticker, ticker): ticker for ticker in tickers}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tickers), desc="Saving Raw Data"):
                ticker = futures[future]
                try:
                    success, msg = future.result()
                    if not success and msg:
                        tqdm.write(msg)
                except Exception as e:
                    tqdm.write(f"Exception saving raw data for {ticker}: {e}")

        return

    def _save_single_raw_ticker(self, ticker: str):
        """Worker method for multi-threaded raw data saving."""
        try:
            df = self.df_data[ticker].dropna(how='all')
            if df.empty:
                err = f"Warning: Downloaded data for {ticker} is empty. This may indicate a download issue. Skipping saving."
                return False, err
            
            filename = self.stocks[ticker].filename if ticker in self.stocks else ticker.replace("/", "_")
            file_path = os.path.join(self.raw_folder, f"{filename}.csv")
            df.to_csv(file_path)
            return True, ""
        except Exception as e:
            return False, f"Error saving raw data for {ticker}: {e}"

    def _process_single_ticker(self, ticker: str):
        """Worker method for multi-threaded feature processing."""
        try:
            df = self.df_data[ticker].dropna(how='all').copy()
            if df.empty:
                return False, f"Warning: Downloaded data for {ticker} is empty. Skipping."

            sector_str = self.stocks[ticker].sector if ticker in self.stocks else SP500Sector.OTHER
            df = generate_features(df, sector_str=sector_str, is_training=True)
            df.dropna(inplace=True)
            
            if df.empty:
                return False, f"Warning: Processed data for {ticker} is empty after dropping NaNs. Skipping."

            file_path = os.path.join(self.processed_folder, f"{self.stocks[ticker].filename}.csv")
            df.to_csv(file_path)
            return True, ""
        except Exception as e:
            return False, f"Error processing {ticker}: {e}"

    def preProcessing(self):
        """Calculates features and target, then partitions data by stock and saves it."""
        print(f"Processing features and targets, saving to {self.processed_folder}...")
        if self.df_data.empty:
            err = "self.df_data is empty. Ensure that fetch_data() has been called successfully before processing."
            raise ValueError(err)
        
        os.makedirs(self.processed_folder, exist_ok=True)
        
        # Clear existing processed data files
        for f in os.listdir(self.processed_folder):
            if f.endswith(".csv"):
                os.remove(os.path.join(self.processed_folder, f))
                
        tickers = list(self.stocks.keys())
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all ticker processing tasks to the thread pool
            futures = {executor.submit(self._process_single_ticker, ticker): ticker for ticker in tickers}
            
            # Wrap the as_completed iterator with tqdm for a dynamic progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tickers), desc="Processing Features"):
                ticker = futures[future]
                try:
                    success, msg = future.result()
                    if not success and msg:
                        tqdm.write(msg)
                except Exception as e:
                    tqdm.write(f"Exception processing {ticker}: {e}")

        # Clear the massive raw DataFrame from RAM now that processing is done
        self.df_data = pd.DataFrame()
        return

    def load_raw_data(self, isProcessData: bool = True) -> None:
        """Loads raw data from CSV files in rawStock."""
        
        # Load stock metadata if available
        stock_list_path = os.path.join(self.data_folder, "stock_list.csv")
        if not os.path.exists(stock_list_path):
            err = f"stock_list.csv not found in {self.data_folder}. Cannot load stock metadata."
            raise FileNotFoundError(err)

        print(f"Loading stock metadata from {stock_list_path}...")
        df_stocks = pd.read_csv(stock_list_path)
        for _, row in df_stocks.iterrows():
            stock = Stock.from_pd_series(row)
            self.stocks[stock.ticker] = stock

        dfs = []
        if not os.path.exists(self.raw_folder):
            err = f"rawStock folder not found at {self.raw_folder}. Run fetch_data()."
            raise FileNotFoundError(err)
        
        if not isProcessData:
            if not os.path.exists(self.processed_folder):
                err = f"processedStock folder not found at {self.processed_folder}. Run fetch_data()."
                raise FileNotFoundError(err)
            else:
                return
        
        print(f"Loading data from {self.raw_folder}...")
        for stock in self.stocks.values():
            filename = stock.filename
            try:
                file_path = os.path.join(self.raw_folder, f"{filename}.csv")
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                # Reconstruct MultiIndex columns: (Ticker, Price)
                df.columns = pd.MultiIndex.from_product([[stock.ticker], df.columns])
                dfs.append(df)
            except Exception as e:
                raise ValueError(f"Error loading {filename}: {e}")
        
        if not dfs:
            err = f"No valid raw data files found in {self.raw_folder}. Run fetch_data()."
            raise ValueError(err)
        
        # Concatenate all dataframes
        self.df_data = pd.concat(dfs, axis=1)
        self.df_data.sort_index(inplace=True)
        self.df_data.sort_index(axis=1, inplace=True) # Ensure columns (Tickers) are sorted alphabetically
        
        self.preProcessing()
        return

    def fillRelatedTickers(self, central_ticker: str):
        """
        Fetches suppliers and customers for a given central ticker using Finnhub API.
        Dynamically finds US-listed equivalents for non-US stocks.
        """
        api_key = self.finnhub_api_key
        if not api_key:
            print("Warning: Finnhub API Key not set. Skipping fetching related tickers.")
            return

        search_ticker = central_ticker
        # For non-US stocks, try to find a US-listed equivalent (ADR/OTC) for better Finnhub compatibility
        if ".TW" in central_ticker and central_ticker in self.stocks:
            company_name = self.stocks[central_ticker].company
            if company_name:
                print(f"'{central_ticker}' is a non-US ticker. Searching for a US-listed equivalent for '{company_name}'...")
                try:
                    # Clean the company name for better search matching
                    clean_name = company_name.split(',')[0].replace(" Corp.", "").replace(" Inc.", "").replace(" Ltd.", "").replace(" Co.", "").strip()
                    
                    # Finnhub's backend crashes (422 error) on certain complex or multi-word queries.
                    # We take just the first word (e.g. "Uni-President") to bypass the server-side bug.
                    simple_query = clean_name.split()[0] if clean_name else ""
                    search_url = 'https://finnhub.io/api/v1/search'
                    r_search = requests.get(search_url, params={'q': simple_query}, headers={'X-Finnhub-Token': api_key})
                    r_search.raise_for_status()
                    search_results = r_search.json().get('result', [])

                    found_us_ticker = None
                    for item in search_results:
                        symbol = item.get('symbol')
                        # Heuristic: US tickers usually don't have a '.' in them.
                        if symbol and '.' not in symbol and symbol != central_ticker:
                            found_us_ticker = symbol
                            break # Take the first likely US match

                    if found_us_ticker:
                        search_ticker = found_us_ticker
                        print(f"Found US equivalent via Finnhub: '{search_ticker}'. Using it for supply chain lookup.")
                    else:
                        print(f"No direct US listing found via Finnhub. Using original ticker '{central_ticker}'.")

                except requests.exceptions.RequestException as e:
                    print(f"Warning: Finnhub search for US listing failed: {e}")

        print(f"Fetching related tickers (suppliers/customers) for {central_ticker} (using Finnhub with '{search_ticker}')...")

        try:
            url = 'https://finnhub.io/api/v1/stock/supply-chain'
            r = requests.get(url, params={'symbol': search_ticker}, headers={'X-Finnhub-Token': api_key})
            r.raise_for_status() # Raise an exception for bad status codes
            data = r.json().get('data', {})
            
            if not data or not data.get('supplyChain'):
                print(f"No supply chain data returned from Finnhub for '{search_ticker}'.")
                return

            suppliers = []
            customers = []
            all_related_tickers = []

            for item in data.get('supplyChain', []):
                symbol = item.get('symbol')
                if not symbol:
                    continue
                
                all_related_tickers.append(symbol)
                if item.get('key') == 'customer':
                    customers.append(symbol)
                elif item.get('key') == 'supplier':
                    suppliers.append(symbol)
            
            # Store this relationship in the central stock object
            if central_ticker in self.stocks:
                self.stocks[central_ticker].suppliers = sorted(list(set(suppliers)))
                self.stocks[central_ticker].customers = sorted(list(set(customers)))
                print(f"  - Found Suppliers: {self.stocks[central_ticker].suppliers}")
                print(f"  - Found Customers: {self.stocks[central_ticker].customers}")

            for ticker in set(all_related_tickers): # Use set to avoid duplicates
                if ticker in self.stocks:
                    continue
                
                ticker_info = get_yfinance_ticker_info(ticker)
                if ticker_info:
                    # Handle potential sector mapping for non-US stocks if needed
                    if ".TW" in ticker_info["symbol"]:
                        ticker_info["sector"] = SP500Sector().fromYfinanceSector(ticker_info.get("sector", ""))
                    
                    self.stocks[ticker_info["symbol"]] = Stock.from_dict(ticker_info)
                    print(f"Added related stock: {ticker_info['symbol']} ({ticker_info.get('longname', '')})")

        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch supply chain for '{search_ticker}' from Finnhub: {e}")
        
        return

    # --- Two-Stage Transfer Learning Helpers ---

    def loadProcessedData(self, ticker: str=None) -> pd.DataFrame:
        """
        Reads and returns DataFrame directly from processedStock folder.
        """
        print(f"Loading processed data for Ticker: [{ticker}]...")
        
        if ticker is None:
            raise ValueError("Ticker must be provided for particular stage.")

        file_path = os.path.join(self.processed_folder, f"{self.stocks[ticker].filename}.csv")
        if not os.path.exists(file_path):
            raise ValueError(f"Processed data for {ticker} not found at {file_path}")

        return pd.read_csv(file_path, index_col=0, parse_dates=True)

    def _prepare_training_data(self, df_input: pd.DataFrame, seq_length: int = 30, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Helper to convert DataFrames (Wide or Single) into X (Features) and y (Target).
        Target is defined as: Percentage Return (Next Close - Current Close) / Current Close.
        """
        if df_input.empty:
            raise ValueError("Input DataFrame is empty.")
        
        if 'Target' not in df_input.columns:
            raise ValueError("Target column is missing. Ensure data was processed during fetch.")

        if not self.feature_cols:
            raise ValueError("Feature columns not defined. Ensure getStage1Data has been called.")

        missing_features = [c for c in self.feature_cols if c not in df_input.columns]
        if missing_features:
            err = f"Missing required features: {missing_features}"
            raise ValueError(err)
        
        # --- Memory Optimization: Pre-allocate numpy arrays to avoid 2x RAM spike ---
        total_seqs = max(0, len(df_input) - seq_length + 1)
            
        if total_seqs == 0:
             raise ValueError(f"Not enough data to create sequences of length {seq_length}")

        # Allocate memory directly as float32
        X = np.empty((total_seqs, seq_length, len(self.feature_cols)), dtype=np.float32)
        y = np.empty(total_seqs, dtype=np.float32)
        
        idx = 0
        df_sorted = df_input.sort_index()
        dates = df_sorted.index
        features = df_sorted[self.feature_cols].values.astype(np.float32)
        targets = df_sorted['Target'].values.astype(np.float32)
        
        for i in range(len(features) - seq_length + 1):
            X[idx] = features[i:i + seq_length]
            y[idx] = targets[i + seq_length - 1]
            idx += 1
            
        if verbose:
            print(f"Data prepared: {len(X)} sequences of length {seq_length} from {dates.min().date()} to {dates.max().date()}")
        
        return X, y

    def loadGeneralModel(self) -> bool:
        """Loads the General model."""
        model_path = os.path.join(self.data_folder, "general.keras")
        if TransferLearningModel.exists(model_path):
            print(f"Loading General model from {model_path}...")
            self.general_model = TransferLearningModel(input_dim=len(self.feature_cols), seq_length=30)
            self.general_model.load(model_path)
            return True
        
        print(f"General model not found at {model_path}.")
        return False

    def loadMTAGeneralModel(self, loss_fn: str = "ce") -> bool:
        """Loads the MTA General model."""
        model_path = os.path.join(self.data_folder, "mta_general.pt")
        if MTAStockModel.exists(model_path):
            print(f"Loading MTA General model from {model_path}...")
            self.mta_general_model = MTAStockModel(input_dim=len(self.feature_cols), seq_length=30, loss_fn=loss_fn)
            self.mta_general_model.load(model_path)
            return True
        print(f"MTA General model not found at {model_path}.")
        return False

    def loadParticularModel(self, ticker: str) -> bool:
        """Loads the Particular model."""
        safe_ticker = ticker.replace("/", "_")
        model_path = os.path.join(self.data_folder, f"{safe_ticker}.keras")
        if TransferLearningModel.exists(model_path):
            print(f"Loading Particular model from {model_path}...")
            # Create a model instance and load the saved state. The `load` method
            # will update the instance's input_dim to match the saved model.
            loaded_model = TransferLearningModel()
            loaded_model.load(model_path)

            # Verify that the loaded model's features match the current configuration
            config_dim = len(self.feature_cols)
            if loaded_model.input_dim != config_dim:
                raise ValueError(
                    f"Model feature mismatch for '{ticker}'.\n"
                    f"The saved model '{model_path}' was trained with {loaded_model.input_dim} features, "
                    f"but the current configuration uses {config_dim} features.\n"
                    "Please delete the old model file (and its '.scaler' file), then run the full training pipeline "
                    "(e.g., practice_transfer_learning with isTrainModel=True) to generate an up-to-date model."
                )

            self.particular_model = loaded_model
            return True
        print(f"Particular model not found at {model_path}.")
        return False

    def loadMTAParticularModel(self, ticker: str) -> bool:
        """Loads the MTA Particular model."""
        safe_ticker = ticker.replace("/", "_")
        model_path = os.path.join(self.data_folder, f"mta_{safe_ticker}.pt")
        if MTAStockModel.exists(model_path):
            print(f"Loading MTA Particular model from {model_path}...")
            self.mta_particular_model = MTAStockModel(input_dim=len(self.feature_cols), seq_length=30)
            self.mta_particular_model.load(model_path)
            return True
        print(f"MTA Particular model not found at {model_path}.")
        return False

    def trainGeneral(self, epochs=20, isFromScratch=True, patience=3):
        print("\n=== General Pre-training (Out-of-Core Iterative) ===")
        
        files = [f"{stock.filename}.csv" for stock in self.stocks.values()]
        if not files:
            raise ValueError("self.stocks is empty.")

        self.particular_model = None

        if isFromScratch:
            print("Training from scratch. Any existing General model will be overwritten.")
            self.general_model = TransferLearningModel(input_dim=len(self.feature_cols), seq_length=30, hidden_layers=(64, 32))
        else:
            if self.general_model is None:
                self.loadGeneralModel()

        print("Pass 1: Incrementally fitting scaler across all disk files...")
        def _prep_for_scaler(f):
            df = pd.read_csv(os.path.join(self.processed_folder, f), index_col=0, parse_dates=True)
            X, _ = self._prepare_training_data(df, seq_length=self.general_model.seq_length, verbose=False)
            return X
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(_prep_for_scaler, f): f for f in files}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Fitting Scaler"):
                self.general_model.update_scaler(future.result())
                del futures[future] # Free memory immediately!

        best_val_mae = float('inf')
        patience_counter = 0
        save_path = os.path.join(self.data_folder, "general.keras")

        print("Pass 2: Training model sequentially by ticker...")
        for epoch in range(1, epochs + 1):
            print(f"--- Global Epoch {epoch}/{epochs} ---")
            epoch_val_mae = 0.0
            val_count = 0
            
            for stock in self.stocks.values():
                df = self.loadProcessedData(stock.ticker)
                X, y = self._prepare_training_data(df, seq_length=self.general_model.seq_length, verbose=False)
                
                split = int(len(X) * 0.8)
                X_train, y_train = X[:split], y[:split]
                X_test, y_test = X[split:], y[split:]
                
                if len(X_train) == 0:
                    continue
                
                self.general_model.train(X_train, y_train, epochs=1, fit_scaler=False, verbose=True, patience=0)
                
                if len(X_test) > 0:
                    mae = self.general_model.evaluate(X_test, y_test)
                    epoch_val_mae += mae
                    val_count += 1
        
            if val_count > 0:
                avg_val_mae = epoch_val_mae / val_count
                print(f"  -> Average Validation MAE: {avg_val_mae:.4f}")
                
                if avg_val_mae < best_val_mae:
                    best_val_mae = avg_val_mae
                    patience_counter = 0
                    self.general_model.save(save_path)
                    print(f"  -> Model improved. Saved to {save_path}")
                else:
                    patience_counter += 1
                    print(f"  -> No improvement. Patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print("  -> Early stopping triggered!")
                        break

        return

    def trainParticular(self, ticker: str=None, epochs=20, isFromGeneral=True):
        print(f"\n=== Particular Fine-tuning ({ticker}) ===")
        if ticker is None:
            raise ValueError("Ticker must be provided for Particular training.")

        if isFromGeneral:
            self.loadGeneralModel()
            self.particular_model = self.general_model.copy()
            self.general_model = None # Free up RAM by dereferencing the general model
        else:
            self.loadParticularModel(ticker=ticker)

        if self.particular_model is None:
            raise ValueError("General model not loaded.")
        
        X, y = self._prepare_training_data(self.loadProcessedData(ticker), seq_length=self.particular_model.seq_length)
        
        # Split into Train (95%) and Test (5%) so the fine-tuned model sees recent trends
        split = int(len(X) * 0.95)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        self.particular_model.train(X_train, y_train, epochs=epochs, fit_scaler=False)
        print(f"Particular MAE (Train): {self.particular_model.evaluate(X_train, y_train):.4f}")
        print(f"Particular MAE (Test):  {self.particular_model.evaluate(X_test, y_test):.4f}")
        
        save_path = os.path.join(self.data_folder, f"{self.stocks[ticker].filename}.keras")
        self.particular_model.save(save_path)
        print(f"Saved Particular model to {save_path}")
        return

    def trainMTAGeneral(self, epochs: int = 20, loss_fn: str = "ce", patience: int = 3):
        """General pre-training with the MTA model."""
        print(f"\n=== MTA General Pre-training (Out-of-Core, loss={loss_fn}) ===")
        
        files = [f for f in os.listdir(self.processed_folder) if f.endswith(".csv")]
        if not files:
            raise ValueError("No processed data found.")

        tmp = MTAStockModel(input_dim=len(self.feature_cols), seq_length=30, loss_fn=loss_fn)
        self.mta_particular_model = None
        self.mta_general_model = MTAStockModel(input_dim=len(self.feature_cols), seq_length=30, loss_fn=loss_fn)

        print("Pass 1: Incrementally fitting scaler...")
        def _prep_for_scaler_mta(f):
            df = pd.read_csv(os.path.join(self.processed_folder, f), index_col=0, parse_dates=True)
            X, _ = self._prepare_training_data(df, seq_length=tmp.seq_length, verbose=False)
            return X
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(_prep_for_scaler_mta, f): f for f in files}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Fitting Scaler (MTA)"):
                self.mta_general_model.update_scaler(future.result())
                del futures[future] # Free memory immediately!

        best_val_mae = float('inf')
        patience_counter = 0
        save_path = os.path.join(self.data_folder, "mta_general.pt")

        print("Pass 2: Training MTA model sequentially...")
        for epoch in range(1, epochs + 1):
            print(f"--- Global Epoch {epoch}/{epochs} ---")
            epoch_val_mae = 0.0
            val_count = 0
            
            for file in files:
                print(f"Training on {file}...")
                df = pd.read_csv(os.path.join(self.processed_folder, file), index_col=0, parse_dates=True)
                X, y = self._prepare_training_data(df, seq_length=tmp.seq_length, verbose=False)
                
                split = int(len(X) * 0.8)
                X_train, y_train = X[:split], y[:split]
                X_test, y_test = X[split:], y[split:]
                if len(X_train) == 0: continue
                
                self.mta_general_model.train(X_train, y_train, epochs=1, fit_scaler=False, verbose=True, patience=0)
                
                if len(X_test) > 0:
                    mae = self.mta_general_model.evaluate(X_test, y_test)
                    epoch_val_mae += mae
                    val_count += 1
                    
            if val_count > 0:
                avg_val_mae = epoch_val_mae / val_count
                print(f"  -> Average Validation MAE: {avg_val_mae:.4f}")
                
                if avg_val_mae < best_val_mae:
                    best_val_mae = avg_val_mae
                    patience_counter = 0
                    self.mta_general_model.save(save_path)
                    print(f"  -> Model improved. Saved to {save_path}")
                else:
                    patience_counter += 1
                    print(f"  -> No improvement. Patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print("  -> Early stopping triggered!")
                        break

        save_path = os.path.join(self.data_folder, "mta_general.pt")
        self.mta_general_model.save(save_path)
        print(f"Saved MTA General model to {save_path}")
        print("Reloading best MTA General model weights from disk...")
        self.loadMTAGeneralModel(loss_fn=loss_fn)
        return

    def trainMTAParticular(self, ticker: str = None, epochs: int = 20):
        """Particular fine-tuning with the MTA model."""
        print(f"\n=== MTA Particular Fine-tuning ({ticker}) ===")
        if ticker is None:
            raise ValueError("ticker must be provided.")
        if self.mta_general_model is None:
            raise ValueError("MTA General model not initialised. Run trainMTAGeneral first.")

        df_particular = self.loadProcessedData(ticker)
        X, y = self._prepare_training_data(df_particular, seq_length=self.mta_general_model.seq_length)
        split = int(len(X) * 0.95)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.mta_particular_model = self.mta_general_model.copy()
        self.mta_particular_model.train(X_train, y_train, epochs=epochs, fit_scaler=False)
        print(f"MTA Particular MAE (Train): {self.mta_particular_model.evaluate(X_train, y_train):.4f}")
        print(f"MTA Particular MAE (Test):  {self.mta_particular_model.evaluate(X_test,  y_test):.4f}")

        save_path = os.path.join(self.data_folder, f"mta_{self.stocks[ticker].filename}.pt")
        self.mta_particular_model.save(save_path)
        return

    def trainMTAModel(
        self,
        ticker: str = None,
        epochs_general: int = 20,
        epochs_particular: int = 20,
        loss_fn: str = "ce",
    ):
        if epochs_general > 0:
            self.trainMTAGeneral(epochs=epochs_general, loss_fn=loss_fn)
        else:
            self.loadMTAGeneralModel(loss_fn=loss_fn)

        if ticker is None:
            print("No ticker specified. Skipping MTA Particular stage.")
            return

        if epochs_particular > 0:
            self.trainMTAParticular(ticker=ticker, epochs=epochs_particular)
        else:
            self.loadMTAParticularModel(ticker=ticker)
        
        return

    def trainModel(self, ticker=None, epochs_general=20, epochs_particular=20):
        if ticker is None:
            err = "No ticker specified. Please provide a ticker for training."
            raise ValueError(err)
            
        if epochs_general > 0:
            self.trainGeneral(epochs=epochs_general)
        
        if epochs_particular > 0:
            self.trainParticular(ticker=ticker, epochs=epochs_particular, isFromGeneral=True)
            
        return

    def predict_day(self, ticker: str=None, date: str=datetime.datetime.now().strftime("%Y-%m-%d")):
        """
        Predicts the next day's closing price for the given ticker using the best available model.
        """
        if ticker is None:
            raise ValueError("Ticker must be provided for prediction.")
        
        # 1. Check and Load Model
        if not self.loadParticularModel(ticker=ticker):
            err = f"No trained model found for ticker [{ticker}]. Please run trainParticular first."
            raise ValueError(err)
            
        # 2. Download fresh data
        print(f"Fetching fresh data...")
        
        target_dt = datetime.datetime.strptime(date, "%Y-%m-%d")
        end_dt = target_dt - datetime.timedelta(days=1)
        start_dt = end_dt - datetime.timedelta(days=180)
        
        # Download sufficient history for rolling windows (e.g., 6 months)
        df_ticker = yf.download(ticker, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), auto_adjust=True, progress=True)
        
        if df_ticker.empty:
            err = f"No data found for {ticker} on yfinance."
            raise ValueError(err)

        # 3. Feature Engineering
        sector_str = self.stocks[ticker].sector if ticker in self.stocks else SP500Sector.OTHER
        df_ticker = generate_features(df_ticker, sector_str=sector_str, is_training=False)
        
        # 4. Prepare Input
        seq_length = 30
        df_ticker = df_ticker.dropna()
        if len(df_ticker) < seq_length:
            raise ValueError(f"Not enough data to fetch {seq_length} days for prediction.")

        last_row = df_ticker.iloc[[-1]]
        last_close = float(last_row['Close'].values.flatten()[0])
        last_date = last_row.index[0].date()
        
        last_window = df_ticker.iloc[-seq_length:]
        X_new = last_window[self.feature_cols].values
        X_new = np.expand_dims(X_new, axis=0) # Shape: (1, 30, num_features)
        
        # 5. Predict
        pred_pct_return = self.particular_model.predict(X_new)[0][0]
        pred_price = last_close * (1 + pred_pct_return)
        
        print(f"Prediction for {ticker} (Ticker Model) on {last_date}:")
        print(f"  Last Close: {last_close:.2f}")
        print(f"  Predicted Return: {pred_pct_return*100:.2f}%")
        print(f"  Predicted Next Close: {pred_price:.2f}")
        
        return pred_price

    def predict_day_mta(self, ticker: str = None, date: str = datetime.datetime.now().strftime("%Y-%m-%d")):
        """
        Predicts the next day's direction for the given ticker using the MTA model.
        Returns the predicted pseudo-return signal (bull - bear probability).
        """
        if ticker is None:
            raise ValueError("Ticker must be provided.")

        if not self.loadMTAParticularModel(ticker=ticker):
            raise ValueError(f"No trained MTA model for [{ticker}]. Run trainMTAModel first.")

        print("Fetching fresh data for MTA prediction...")
        target_dt = datetime.datetime.strptime(date, "%Y-%m-%d")
        end_dt    = target_dt - datetime.timedelta(days=1)
        start_dt  = end_dt   - datetime.timedelta(days=180)

        df_ticker = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=True, progress=False
        )
        if df_ticker.empty:
            raise ValueError(f"No data found for {ticker} on yfinance.")

        sector_str = self.stocks[ticker].sector if ticker in self.stocks else SP500Sector.OTHER
        df_ticker = generate_features(df_ticker, sector_str=sector_str, is_training=False).dropna()
        seq_length = self.mta_particular_model.seq_length

        if len(df_ticker) < seq_length:
            raise ValueError(f"Not enough data ({len(df_ticker)} rows) for seq_length={seq_length}.")

        last_row   = df_ticker.iloc[[-1]]
        last_close = float(last_row["Close"].values.flatten()[0])
        last_date  = last_row.index[0].date()

        X_new = df_ticker.iloc[-seq_length:][self.feature_cols].values
        X_new = np.expand_dims(X_new, axis=0)   # (1, seq_length, n_features)

        # Class probabilities
        probs  = self.mta_particular_model.predict_proba(X_new)[0]  # [bear, sideways, bull]
        signal = float(probs[2] - probs[0])                     # bull - bear

        pred_class = ["Bear", "Sideways", "Bull"][int(np.argmax(probs))]

        print(f"\nMTA Prediction for {ticker} on {last_date}:")
        print(f"  Last Close       : {last_close:.2f}")
        print(f"  Bear  probability: {probs[0]*100:.1f}%")
        print(f"  Side  probability: {probs[1]*100:.1f}%")
        print(f"  Bull  probability: {probs[2]*100:.1f}%")
        print(f"  Predicted class  : {pred_class}")
        print(f"  Signal (bull-bear): {signal:+.4f}")

        return signal
