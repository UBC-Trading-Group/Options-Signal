# OptionSignal

## data description

* opprcdXXXX.csv: These files contain options for several indexes/stocks (SPX, NDX, MSFT, and KO) from 1996 to 2009. They were extracted from [here](https://www.dropbox.com/scl/fi/bq4h5ezzlebkh9evnd052/optionmWRDS_2010_05.zip?dl=0&oref=e&r=AB4WoH1qJwJ2ij0XMcuWbYxqiB66L6S8Rx-6VyWRhZdwIadT8o4WcDj-dCgQp44ys1rQ4Cysy_3BrG5DYqC_YjxENbMuaoDsCzFiVPnvLumd-HbkOzjjsdvZ-hPM0-PQQU8oCEtOPclskhjTYM18_TxTRF0CGbx9I7c073hRh7wRdjxUevLGGVSoHP3K9YPYGZE&sm=1), which were originally downloaded from [WRDS optionMetrics](https://wrds-www.wharton.upenn.edu/pages/get-data/optionmetrics-trial/ivy-db-us/options/option-prices/).

* riskFreeRate.csv: This was downloaded from [WRDS Treasuries/ Riskfree Series (1-month and 3-month)](https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/annual-update/treasuries/riskfree-series-1-month-and-3-month/). It contains the annualized yield for 3-month US T-bills from 1996 to 2022. The variable description could be found in the 'Variable Description' tab in the download page.

* stock_mock_universe_4_stocks.csv: This was downloaded from [WRDS Stock / Security files](https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/annual-update/stock-security-files/daily-stock-file/). It contains the daily stock prices from 1996 to 2022 for XOM, CVX, LLY, JNJ. The variable description could be found in the 'Variable Description' tab in the download page.

* stock_mock_universe_6_stocks.csv: This was downloaded from [WRDS Stock / Security files](https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/annual-update/stock-security-files/daily-stock-file/). It contains the daily stock prices from 1996 to 2022 for AAPL, MSFT, XOM, CVX, LLY, JNJ. The variable description could be found in the 'Variable Description' tab in the download page.

* vix.csv: VIX downloaded from WRDS

## script description

* cal_VIX_sample.py: sample code for calculate the VIX based on the near-term options

* ImpliedReturnLib.py: functions for calculating imiplied equilibrium returns.
