import pandas_datareader.data as web
import yfinance as yf
from yahoofinancials import YahooFinancials
import pandas as pd
import numpy as np
import lxml as lx

# %% financial info using web and get_data_yahoo
price = web.get_data_yahoo("GRG.L", "max")["Adj Close"]
price
price.plot()

# %% financial info using yfinance
greggs = yf.Ticker("GRG.L")
greggs
greggs.info
price_history = greggs.history(period="max")
# %% financial info using yahoofinancials
ticker = "GRG.L"
yahoo_financials = YahooFinancials(ticker)

balance_sheet_data_qt = yahoo_financials.get_financial_stmts("yearly", "balance")
balance_sheet_data_qt
# %%
