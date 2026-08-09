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
        price = quote.get("regularMarketPrice")
        # A handful of listings (e.g. delisted/inactive tickers) have no live
        # price at all - skip those instead of crashing on the missing field.
        if price is None:
            continue
        if "-" not in quote["symbol"] and 1 <= price <= 50:
            symbols.append(quote["symbol"])
            quote_info[quote["symbol"]] = {
                "MarketCap": quote.get("marketCap"),
                "Price": price,
            }

tickers_string = " ".join(symbols)
columns = ["Ticker", "MarketCap", "Price", "Histogram", "Derivative", "RSI"]
potentials = pd.DataFrame(columns=columns)
print(len(symbols))

cached_data = (
    "cached_stock_data_"
    + interval
    + "_"
    + period
    + "_"
    + datetime.datetime.now().strftime("%Y%m%d%H")
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

        # 14-period RSI (Wilder's smoothing, approximated via EMA with alpha=1/14)
        # - measures how overbought/oversold the stock is; 0-100, with >70
        # typically read as overbought, <30 oversold.
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))

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
                "RSI": rsi.iat[-1],
            }
            potentials.loc[len(potentials)] = new_row

print(potentials.shape[0])
potentials.to_csv("potentials" + "_" + interval + "_" + period + ".csv", index=False)
