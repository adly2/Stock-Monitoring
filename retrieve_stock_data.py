import requests
import pandas as pd
import yfinance as yf

# base_url = 'https://financialmodelingprep.com/api/v3'
# apiKey = "pQgnXt0XoOsnuUkMPsvstpeOtuefLJ9H"

# list_url = f'{base_url}/symbol/TSX?apikey={apiKey}'

# response = requests.get(list_url)
# data = response.json()
# print(data[0])
# df = pd.DataFrame(data)
# df.sort_values(by='marketCap',ascending=False, inplace=True)
# df = df[df["price"] < 50]
# df_no_duplicates = df.drop_duplicates(subset=["name"], keep="first")
# df_filtered = df_no_duplicates[~df_no_duplicates["symbol"].str.contains("-")]
# print(df_filtered['symbol'])

cq = yf.EquityQuery('eq', ['region', 'ca'])
screener = yf.Screener()
screener.set_default_body(query=cq)
screener.patch_body({"size": 250})
result = screener.response
print(result['total'])
symbols = []

# for git

for i in range(0, 3250, 250):
    screener.patch_body({"offset": i})
    result = screener.response
    symbols.extend(
        quote['symbol'] for quote in result['quotes'] 
        if '-' not in quote['symbol'] and 1 <= quote['regularMarketPrice'] <= 50
    )

tickers_string = ' '.join(symbols)
tickers = yf.Tickers(tickers_string)
columns = ["Ticker", "Histogram", "Derivative"]
potentials = pd.DataFrame(columns=columns)
print(len(symbols))
print(symbols.index('SPFD.TO'))
print(tickers.tickers['SPFD.TO'].info)

for ticker in symbols:
    print(ticker)
    df = tickers.tickers[ticker].history(period='3mo', interval='1d')
    if not df.empty:
        ema_12 = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
        MACD = ema_12-ema_26
        signal = MACD.ewm(span=9, adjust=False, min_periods=9).mean()
        histogram = MACD - signal
        derivative = histogram.iat[-1] - histogram.iat[-2]
        if histogram.iat[-1] <= 0.5 and histogram.iat[-1] >= -0.5 and derivative >= 0.3:
            new_row = {'Ticker': ticker, 'Histogram': histogram.iat[-1], 'Derivative': derivative}
            potentials.loc[len(potentials)] = new_row

print(potentials.shape[0])

# 
# i = 0
# j = 0
# ticker = df["symbol"][i]


# while j < 5:
#     if df["name"][i] not in stock_data["Name"].values:
#         ticker = df["symbol"][i]
#         stock_url = f'{base_url}/historical-price-full/{ticker}?apikey={apiKey}'
#         response = requests.get({})
#         stock_data = stock_data.append({"Name": df["name"][i], "MACD - SIGNAL": None, "Increasing": None}, ignore_index=True)
#         j += 1
#     i += 1
# print(i)
