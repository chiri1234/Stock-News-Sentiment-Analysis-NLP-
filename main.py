from urllib.request import urlopen, Request
from urllib.error import HTTPError
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'GOOG', 'META', 'AAPL', 'TSLA']

news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})

    try:
        response = urlopen(req)
        html = BeautifulSoup(response, features='html.parser')
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table
    except HTTPError as e:
        print(f"Failed to retrieve data for {ticker}: {e}")

parsed_data = []

for ticker, news_table in news_tables.items():
    if news_table:
        for row in news_table.findAll('tr'):
            title = row.a.text
            date_data = row.td.text.split(' ')

            if len(date_data) == 1:
                time = date_data[0]
                date = None
            else:
                date = date_data[0]
                time = date_data[1]

            parsed_data.append([ticker, date, time, title])

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

# Handle date format and fill NaT with today's date
df['date'] = pd.to_datetime(df['date'], format='%b %d', errors='coerce').dt.date
df['date'] = df['date'].fillna(pd.to_datetime('today').date())

# Initialize VADER for sentiment analysis
vader = SentimentIntensityAnalyzer()

# Calculate compound sentiment score
df['compound'] = df['title'].apply(lambda title: vader.polarity_scores(title)['compound'])

# Ensure the 'compound' column contains numeric data only
df['compound'] = pd.to_numeric(df['compound'], errors='coerce')

# Group by ticker and date, and calculate the mean of the 'compound' column
mean_df = df.groupby(['ticker', 'date'])['compound'].mean().unstack()

# Define a list of colors for each company
colors = ['blue', 'green', 'red', 'orange', 'purple']

# Plot sentiment scores over different dates as a bar graph with custom colors
if not mean_df.empty:
    mean_df.plot(kind='bar', figsize=(10, 6), color=colors)

    plt.title('Sentiment Analysis for Each Company by Date')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.legend(title='Company')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No data available for plotting sentiment scores.")