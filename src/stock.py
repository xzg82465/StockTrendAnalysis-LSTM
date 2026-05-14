import pandas as pd

class Stock:
    def __init__(self):
        self.exchange: str = ""
        self.ticker: str = ""
        self.sector: str = ""
        self.company: str = ""
        self.business_partners: list[str] = []

        self.filename: str = ""
        return
    
    @classmethod
    def from_dict(cls, data: dict):
        stock = cls()
        stock.exchange = data.get("exchange", "")
        stock.ticker = data.get("symbol", "")
        stock.sector = data.get("sector", "")
        stock.company = data.get("longname", "")
        stock.filename = data.get("filename", "")
        return stock
    
    @classmethod
    def from_pd_series(cls, series: pd.Series):
        stock = cls()
        stock.exchange = series.get("exchange", "")
        stock.ticker = series.get("symbol", "")
        stock.sector = series.get("sector", "")
        stock.company = series.get("longname", "")
        stock.filename = series.get("filename", "")
        return stock

    def to_dict(self):
        return {
            "exchange": self.exchange,
            "symbol": self.ticker,
            "sector": self.sector,
            "longname": self.company,
            "filename": self.filename
        }
    
    def __str__(self):
        return f"Stock(ticker='{self.ticker}', sector='{self.sector}', company='{self.company}', exchange='{self.exchange}')"
