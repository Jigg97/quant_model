#%%
import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from functools import reduce
import os


sector = "storage"

work = r"C:\Users\Joshua.Higgins\Dropbox\Investing\Companies\storage\quants"

work = work.replace(os.sep, "/")

home = "/Users/jjhiggins97/Dropbox/Investing/Companies/storage/quants"

location = work

#%% looping through the financial files of each company in my selected sector and performing the relevant tasks

# Defining all main variables first

# dictionary to convert the CAPIQ exchange codes to the yahoo finance ones

exch_codes = {"LSE": ".L", "NYSE": "", "ENXTBR": ".BR"}

start_year = 2015

companies = []
tickers = []


def quant_model(location):

    for filename in os.listdir(location):

        # removing financials.xls from the file name

        file_nonfin = filename.replace(" Financials.xls", "")

        # splitting each word in the file name into separate ones, using space as the delimiter. Then, taking the final two words needed to create the yf ticker. Changed name to ticker1 so that we can check if the first item (the exchange code) is London. If so, we must divide the price by 100 as we quote our prices in pence.

        ticker1 = file_nonfin.split(" ")[-2:]

        # Need a swap function to swap the positions of the exchange code and ticker to create the full yahoo finance ticker

        def swap(a):
            a[0], a[1] = a[1], a[0]
            return a

        # start inside the inner brackets. I first match the items in the ticker with the exchange code dictionary above, which spits out the yf exchange code. I then use my user-defined swap function from above to swap round the ticker code and new exch code. Finally, I join them together to create the final yf ticker.

        ticker = "".join(swap([exch_codes.get(item, item) for item in ticker1]))

        # Same as before - splitting the filename into separate words, but this time choosing up to, but not including, the final 2 words. This is to get the company name. Then joining to get the final name.

        splitter = file_nonfin.split(" ")[:-2]

        company = " ".join(splitter)

        # Appending to the relevant lists

        tickers.append(ticker)

        companies.append(company)

        # Trying to loop through all files in a directory (sector)

        #%% We open the file manually before using read_excel to stop file size warnings https://github.com/pandas-dev/pandas/  issues/16620

        init_open = xlrd.open_workbook(
            location + "/" + filename, logfile=open(os.devnull, "w"),
        )

        df_is = pd.read_excel(
            init_open,
            sheet_name="Income Statement",
            skiprows=range(1, 14),
            usecols="A:AP",
        )

        df_bs = pd.read_excel(
            init_open, sheet_name="Balance Sheet", skiprows=range(1, 14), usecols="A:AP"
        )

        df_cf = pd.read_excel(
            init_open, sheet_name="Cash Flow", skiprows=range(1, 14), usecols="A:AP"
        )

        df_bs.iloc[0] = df_is.iloc[0]

        def initial_tran(df):

            # Removing the column containing DSC (Data Source Change) as this is not needed

            df = df.drop(columns=df.columns[(df == "DSC").any()])

            # Setting first row as header

            new_header = df.iloc[0]  # grab the first row for the header
            df = df[1:]  # take the data less the header row
            df.columns = new_header  # set the header row as the df header

            # Slicing all crap off the column names so that we just have the date. This takes off the characters up to but not  including index 12
            df.rename(columns=lambda x: x[-11:], inplace=True)

            df = df.rename(columns={"iod Ending\n": "period_ending"})

            # Transpose so I can plot properly

            df = np.transpose(df)

            # Inserting a new index so that all columns can be plotted. That is, pushing all columns to the right by 1

            df.reset_index(level=0, inplace=True)

            # Setting first row as header

            new_header = df.iloc[0]  # grab the first row for the header
            df = df[1:]  # take the data less the header row
            df.columns = new_header  # set the header row as the df header

            # Getting rid of nan columns

            df = df.dropna(axis=1, how="all")
            # Taking leading and trailing spaces from the column names. They must first be recognised as strings before doing this
            df.columns = df.columns.astype(str).str.strip()
            # Converting all column names to lower case
            df.columns = map(str.lower, df.columns)
            # Replacing spaces with '_' in the column names. They must first be recognised as strings before doing this
            df.columns = df.columns.astype(str).str.replace(" ", "_")
            # Setting the date to a datetime type so that it is graphable
            df["period_ending"] = pd.to_datetime(df["period_ending"])

            return df

        # Applying initial_tran function to clean up all financial statements

        df_is = initial_tran(df_is)
        df_cf = initial_tran(df_cf)
        df_bs = initial_tran(df_bs)

        df_is.drop(columns="currency", inplace=True)
        df_cf.drop(columns="currency", inplace=True)

        # Getting price data for the dividend yields

        ticker = yf.Ticker(ticker)

        df_price = ticker.history(period="max")

        # Both adjustments below are stolen from the initial_tran function

        df_price.columns = map(str.lower, df_price.columns)

        df_price.reset_index(level=0, inplace=True)

        df_price = df_price.rename(columns={"Date": "period_ending"})

        #%% df_price does not include weekends, which is a problem when merging with financial  statements as a lot of the FYEs are on weekends. We fix this issue below

        # Using pandas date_range method to get a time series - including weekends - from the df_price  start to end date

        time_series = pd.DataFrame(
            pd.date_range(
                start=df_price["period_ending"].iloc[0],
                end=df_price["period_ending"].iloc[-1],
            )
        )

        time_series.rename(columns={0: "period_ending"}, inplace=True)

        # Merging time series and price dataframe to get the missing weekends

        df_price = df_price.merge(time_series, how="outer", on="period_ending")

        df_price.sort_values("period_ending", inplace=True)

        df_price["close"].ffill(inplace=True)

        df_price.rename(columns={"close": "price"}, inplace=True)

        def divider(df):
            price = df["price"]
            if ticker1[0] == ".L":
                price = price / 100
            return price

        df_price["price"] = divider(df_price)

        # Now need to fill forward for the closing prices

        # Merging all three resource dataframes at once with the reduce function

        dfs = [df_is, df_cf, df_bs, df_price]

        df_fin = reduce(
            lambda left, right: pd.merge(left, right, on="period_ending", how="inner"),
            dfs,
        )

        # I now want a dataframe that just contains the important stuff; a key stats dataframe

        df_stats = df_fin[["period_ending"]].copy()

        # Only taking data from the chosen year onwards

        df_stats = df_stats[df_stats["period_ending"].dt.year >= start_year]

        df_stats["dvd_per_share"] = df_fin["dividends_per_share"]

        # Calculating dividend yield

        df_stats["dvd_yld"] = df_fin["dividends_per_share"] / df_fin["price"]

        # Converting this number to a float64

        df_stats["dvd_yld"] = pd.to_numeric(df_stats["dvd_yld"])

        # Average dividend over the chose time period

        df_stats["dvd_av"] = df_stats["dvd_yld"].mean()

        # Ther percentage change yoy in dividends over the years

        df_stats["dvd_gth"] = df_stats["dvd_per_share"].pct_change()

        years = len(df_stats.index) - 1

        def CAGR(df, metric):

            metric = df[metric]
            first = metric.iloc[0]
            last = metric.iloc[-1]

            return (last / first) ** (1 / years) - 1

        df_stats["dvd_cagr"] = CAGR(df_stats, "dvd_per_share")

        df_stats.insert(loc=0, column="company", value=company)

    return df_stats, companies, tickers


#%%
stats, companies, tickers = quant_model(location)


#%%
sns.set(style="darkgrid")

sns.lineplot(x="period_ending", y="dvd_yld", data=df_stats)


# Have a look at multiple downloads for companies: https://www.quickprogrammingtips.com/python/how-to-download-multiple-files-concurrently-in-python.html


# Finish off looping, then see what's next
# Sort out yahoofinancials problem - need the forward dividend yield. Links to sort it are below
# https://pypi.org/project/yahoofinancials/
# https://github.com/JECSand/yahoofinancials/issues/71
# https://github.com/JECSand/yahoofinancials/pull/72/files#diff-859f0dd652dec0f5114399943a987467R71

# Can get institutional and main holders on yahoo finance - made not be as widely available as CAPIQ though
# Merge all df_stats files with the reduce function
# Sort out units problem (millions vs absolute)
# Need to go through classes to solve this problem:

# class array(list):
#    def swap(self, i, j):
#        self[i], self[j] = self[j], self[i]
#
# test = [1, 2, 34, 3]
#
# test.swap([0, 1])


# %%
