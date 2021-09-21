import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
import os
from Twitter_Tools import *
import statsmodels.api as sm
from statsmodels.tsa.api import VAR


class NLPForecaster:

    def __init__(self):
        self.forecast_df = pd.DataFrame(data=None, columns=[ 'Open','High','Low','Close','Adj Close','Volume',\
                                                             'Aggregate Sentiment','Number of Tweets',\
                                                             'Positive Tweets','Negative Tweets'], dtype='float32')
        self.sentiment_df = pd.DataFrame
        self.price_df = pd.DataFrame
        self.ticker = ''

    def append_df(self,  df):
        temp_dataframe = pd.DataFrame(data=df, columns=[ 'Open','High','Low','Close','Adj Close','Volume',\
                                                             'Aggregate Sentiment','Number of Tweets',\
                                                             'Positive Tweets','Negative Tweets'], dtype='float32')
        self.forecast_df = self.forecast_df.append(temp_dataframe)

    def store_df(self, path, append=True):
        # Store into file, with or without append flag
        if append:
            self.forecast_df.to_csv(path_or_buf=path, mode='a')
        else:
            self.forecast_df.to_csv(path_or_buf=path)

    def fetch_old_df(self, path):
        # Retrieve old data and compile without duplicates
        self.forecast_df = pd.read_csv(filepath_or_buffer=path, index_col='Date')
        self.forecast_df = self.forecast_df[~self.forecast_df.index.duplicated(keep='last')]
        self.forecast_df = self.forecast_df.drop(['Date'], axis=0)
        self.forecast_df.sort_index(inplace=True)

    def build_forecast_df(self, ticker, total_tweets=30000, path=os.getcwd()+'\\AAPL.csv', store=False, append=False):
        self.ticker = ticker
        self.sentiment_df = query_twitter(ticker, total_tweets)
        self.price_df = fetch_7day_price(ticker)
        self.forecast_df = self.price_df.join(self.sentiment_df, how='outer')
        if store:
            self.store_df(path=path, append=append)

    def plot_open_and_twitter(self):
        ax = self.forecast_df['Open'].plot(figsize=(10, 6))
        ax2 = self.forecast_df["Aggregate Sentiment"].plot(secondary_y=True)
        ax.set_ylabel('AAPL Open Price')

        ax.right_ax.set_ylabel('AAPL Twitter Sentiment Score')
        ax.set_title("AAPL Opening Stock Price and Associated Twitter Sentiment Score")
        ax2.legend(loc=4)
        ax.legend()
        return ax, ax2

    def plot_controversy_and_twitter(self):
        if 'Controversy' in self.forecast_df:
            ax = self.forecast_df['Open'].plot(figsize=(10, 6))
            ax2 = self.forecast_df["Controversy"].plot(secondary_y=True)
            ax.set_ylabel('AAPL Open Price')

            ax.right_ax.set_ylabel('AAPL Twitter Controversy Score')
            ax.set_title("AAPL Opening Stock Price and Associated Twitter Controversy Score")
            ax2.legend(loc=4)
            ax.legend()
            return ax, ax2
        else:
            self.forecast_df['Controversy'] = (self.forecast_df['Negative Tweets'] + self.forecast_df['Positive Tweets'])/ self.forecast_df['Positive Tweets']
            ax = self.forecast_df['Open'].plot(figsize=(10, 6))
            ax2 = self.forecast_df["Controversy"].plot(secondary_y=True)
            ax.set_ylabel('AAPL Open Price')

            ax.right_ax.set_ylabel('AAPL Twitter Controverst Score')
            ax.set_title("AAPL Opening Stock Price and Associated Twitter Controversy Score")
            ax2.legend(loc=4)
            ax.legend()
            return ax, ax2

    def build_predictor(self, steps=1):
        # Warning, if Nan present in your dataframe you may need to run an interpolation to make the forecastor function as intended
        # TODO properly merge with DF within function without spaghetti code
        model = VAR(self.forecast_df)
        model_fit = model.fit(maxlags=3)
        return model_fit.forecast(model_fit.y, steps=steps)

def fetch_7day_price(stock_name, days=7):

    return yf.download(stock_name, start=date.today() - timedelta(days=days), end=date.today() + timedelta(days=1))



