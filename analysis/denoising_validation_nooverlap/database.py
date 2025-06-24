import pandas as pd
from sqlalchemy import create_engine, text
class DatabaseUBCTG():
    
    # constructor
    def __init__(self):
        self.engine = create_engine('mysql+pymysql://admin:ubctgquant@ubctg.con7266gcvin.us-east-2.rds.amazonaws.com/ubctg')
    def _get_universe_data(self, column, start , end ):
        conn = self.engine.connect()
        query = f"SELECT * FROM ubctg.{column} WHERE Date >= '{start}' AND Date <= '{end}'"
        df = pd.read_sql(text(query), conn)
        conn.close()
        return df
    def get_returns_universe(self, start = "2000-01-01", end = "2022-12-31"):
        return self._get_universe_data('Returns', start, end)
"""
    # returns a pandas dataframe with price data from "start" to "end" of all stocks in the universe
    def get_price_universe(self, start = "2020-01-01", end = "2022-12-31"):
        return self._get_universe_data('Price', start, end)
    
    # returns a pandas dataframe with volume data from "start" to "end" of all stocks in the universe
    def get_volume_universe(self, start, end):
        return self._get_universe_data('Volume', start, end)
    # returns a pandas dataframe with marketcap data from "start" to "end" of all stocks in the universe
    def get_market_cap_universe(self, start, end):
        return self._get_universe_data('DlyCap', start, end)
     # returns a pandas dataframe with shares outstanding data from "start" to "end" of all stocks in the universe
    def get_shares_outstanding_universe(self, start, end):
        return self._get_universe_data('ShrOut', start, end)
    
    """
