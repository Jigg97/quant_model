from yahoofinancials import YahooFinancials

import yfinance as yf

ticker = "BYG.L"

ticker = yf.Ticker(ticker)

test = ticker.institutional_holders
