import pandas as pd
import yfinance as yf
import datetime
import os
from dotenv import load_dotenv

load_dotenv()  # reads PERIOD/INTERVAL (and anything else) from .env
interval = os.getenv("INTERVAL")
period = os.getenv("PERIOD")

# Screen for Canadian-listed equities via Yahoo Finance's screener API.
# Note: yfinance replaced the old yf.Screener() class with this yf.screen()
# function in newer versions - Screener() no longer exists.
cq = yf.EquityQuery("eq", ["region", "ca"])
symbols = []
# marketCap and regularMarketPrice come from the screener response itself -
# yf.download() below only returns OHLCV history, no market cap or price.
quote_info = {}

# The screener only returns one page (250 results) at a time, so page through
# offsets to collect (up to) 4612 symbols (all Canadian stocks available as of 31st July 2026),
# keeping only stocks priced $1-$50 and excluding tickers
# with a "-" (e.g. warrants, preferred shares, when-issued).
for i in range(0, 4750, 250):
    result = yf.screen(cq, offset=i, size=250)
    for quote in result["quotes"]:
        if "-" not in quote["symbol"] and 1 <= quote["regularMarketPrice"] <= 50:
            symbols.append(quote["symbol"])
            quote_info[quote["symbol"]] = {
                "MarketCap": quote.get("marketCap"),
                "Price": quote["regularMarketPrice"],
            }

tickers_string = " ".join(symbols)
columns = ["Ticker", "MarketCap", "Price", "Histogram", "Derivative"]
potentials = pd.DataFrame(columns=columns)
print(len(symbols))

cached_data = (
    "cached_stock_data_"
    + interval
    + "_"
    + period
    + "_"
    + datetime.datetime.now().strftime("%Y%m%d")
    + ".pkl"
)

if os.path.exists(cached_data):
    data = pd.read_pickle(cached_data)
else:
    # Fetch PERIOD of INTERVAL price data for every symbol in one multi-threaded
    # batch call instead of one HTTP request per ticker
    data = yf.download(
        tickers_string,
        period=period,
        interval=interval,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    # Save to disk so that I can rerun quickly if needed
    data.to_pickle(cached_data)

# For each ticker, compute the MACD histogram and flag stocks where the
# histogram is near zero (about to cross the signal line) and rising -
# a potential bullish MACD crossover setup.
for ticker in symbols:
    df = data[ticker].dropna(how="all")
    if len(df) >= 2:  # need at least 2 rows to compute the histogram's derivative
        ema_12 = df["Close"].ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False, min_periods=26).mean()
        MACD = ema_12 - ema_26
        signal = MACD.ewm(span=9, adjust=False, min_periods=9).mean()
        histogram = MACD - signal
        derivative = histogram.iat[-1] - histogram.iat[-2]  # is the histogram rising?
        if (
            histogram.iat[-1] <= 0.05
            and histogram.iat[-1] >= -0.05
            and derivative >= 0.01
        ):
            new_row = {
                "Ticker": ticker,
                "MarketCap": quote_info[ticker]["MarketCap"],
                "Price": quote_info[ticker]["Price"],
                "Histogram": histogram.iat[-1],
                "Derivative": derivative,
            }
            potentials.loc[len(potentials)] = new_row

print(potentials.shape[0])
potentials.to_csv("potentials" + "_" + interval + "_" + period + ".csv", index=False)
