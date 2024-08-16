import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from pandas_datareader import data 
import yfinance as yfin
from keras.models import load_model
from datetime import datetime
import tensorflow as tf
import os
from dotenv import load_dotenv, dotenv_values
load_dotenv()

# a1
from newsapi.newsapi_client import NewsApiClient
from textblob import TextBlob
 
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential

yfin.pdr_override()
import streamlit as st

#n1
import requests

#c1
st.title(' Stock Trend Prediction. 📈')
api_key = os.getenv("api_key")

st.write("---")
st.subheader('Data Aanalysis')
#user_input
user_input = st.sidebar.text_input(' Enter Stock Ticker : ','MSFT')

default_value_sd=datetime(2020,1,14) 
default_value_ed=datetime(2023,1,14) 
max_allowed_date = datetime(2023,12,31)
start_date=st.sidebar.date_input('start_date',default_value_sd,max_value=max_allowed_date)
end_date=st.sidebar.date_input('end_date',default_value_ed)
df = data.get_data_yahoo(user_input,start_date,end_date) 
#c3 
dff = pd.DataFrame(df)
#theme
# t1
theme = st.sidebar.radio(" Edit Theme : ",("White","Black","Light Black","Light"))
#t2
if theme=="Black":
    page_bg_img = """
    <style>
        [data-testid = "stAppViewContainer" ]{
        background-color : #454545;
    }
    [data-testid= "stAppViewContainer"] div[role="button"],
    [data-testid= "stAppViewContainer"].st-emotion-cache-zt5igj e1nzilvr4{
    color: white !important;
    }
    </style>
    """
    st.markdown(page_bg_img,unsafe_allow_html=True)


#t3
if theme=="White":
    page_bg_img = """
    <style>
        [data-testid = "stAppViewContainer"]{
        background-color : rgba(240,240,240);
        }
    </style>
    """
    st.markdown(page_bg_img,unsafe_allow_html=True)
#t4
if theme=="Light Black":
    page_bg_img = """
    <style>
        [data-testid = "stAppViewContainer"]{
        background-color : rgba(100,100,100);
    }
    </style>
    """
    st.markdown(page_bg_img,unsafe_allow_html=True)
#t5
if theme=="Light":
    page_bg_img = """
    <style>
        [data-testid = "stAppViewContainer"]{
        background-color : #F4E7E7;
    }
    </style>
    """
    st.markdown(page_bg_img,unsafe_allow_html=True)
#data
t1, t2,t3,t4,t5  = st.tabs(['Data','sentimates','BarCharts','News','F&Q'])

with t1:    
    
    import requests
    def get_stock_price(symbol, api_key):
        base_url = "https://www.alphavantage.co/query"
        function = "GLOBAL_QUOTE"
    
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": api_key,
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        if "Global Quote" in data:
            stock_data = data["Global Quote"]
            stock_price = stock_data["05. price"]
            return stock_price
        else:
            return None


    api_key = os.getenv("api_key")
    stock_symbol = user_input  # Replace with the desired stock symbol
# user_input = st.text_input('Enter Stock Ticker','MSFT')

    stock_price = get_stock_price(stock_symbol, api_key)


    if stock_price:
        print(f"The current stock price of {stock_symbol} is ${stock_price}")
        print(stock_symbol)
        st.write(f"The current stock price of {stock_symbol} is ${stock_price}")
    else:
        print(f"Failed to retrieve stock price for {stock_symbol}")
    # st.write(f"Failed to retrieve stock price for {stock_symbol}")

    st.subheader('Data From 2020-2023')
    st.write(dff.describe())

#a2
with t2:
    newsapi = NewsApiClient(api_key='9caa07b5d554424da0090645f2ba21fb')
    st.header(' Stock Prediction based on Sentiments')
    def get_stock_news(symbol, num_articles=5):
        # Using News API to get stock-related news
        headlines = newsapi.get_everything(q=symbol, language='en', sort_by='publishedAt')['articles'][:num_articles]
    
        return [article['title'] for article in headlines]

    def analyze_sentiment(text):
        analysis = TextBlob(text)
    
        # Classify the polarity of the text
        if analysis.sentiment.polarity > 0:
            return "Positive"
        elif analysis.sentiment.polarity == 0:
            return "Neutral"
        else:
            return "Negative"

# Example usage:
    stock_symbol = user_input  # Replace with the stock symbol of your choice
    num_articles = 5

    news_headlines = get_stock_news(stock_symbol, num_articles)

    print(f"\nLatest {num_articles} news headlines for {stock_symbol}:\n")

    for headline in news_headlines:
        st.write(headline)
        sentiment = analyze_sentiment(headline)
        st.write("Sentiment:", sentiment)
        st.write("-" * 50)



# show data to the user 
#c4
with t3:
    st.subheader(' Bar chart of Data From 2020-2023')
    st.bar_chart(df)

#n2 for news
with t4:
    class StockPredictionSystem:
        def __init__(self, news_api_key):
            self.news_api_key = news_api_key

        def get_stock_news(self, stock_symbol, num_articles=5):
        # Replace 'your_news_api_key' with your actual News API key
            api_key = '9caa07b5d554424da0090645f2ba21fb'
            api_endpoint = 'https://newsapi.org/v2/everything'
        
            params = {
                'apiKey': api_key,
                 'q': stock_symbol,
                 'language': 'en',
                 'sortBy': 'publishedAt',
                 'pageSize': num_articles,
             }

            response = requests.get(api_endpoint, params=params)
            data = response.json()

            if 'articles' in data:
                return [article['title'] for article in data['articles']]
            else:
                return []

# Example usage:
    news_api_key = '9caa07b5d554424da0090645f2ba21fb'
    prediction_system = StockPredictionSystem(news_api_key)

    stock_symbol = user_input  # Replace with the stock symbol of your choice
    news_headlines = prediction_system.get_stock_news(stock_symbol)

    st.header(f"Latest news headlines for {stock_symbol}:")
    for headline in news_headlines:
        st.write(headline)
        st.write("-------------------------")

#chatbot 
with t5:
    stock_questions = {
        "What is a stock?": "A stock represents ownership in a company and constitutes a claim on part of the company's assets and earnings.",
        "How does the stock market work?": "The stock market is a platform where buyers and sellers trade shares of publicly listed companies.",
        "What is a dividend?": "A dividend is a distribution of a portion of a company's earnings to its shareholders.",
        "What is market capitalization?":"Market capitalization is the total value of a company's outstanding shares of stock, calculated by multiplying the stock's current market price by the total number of outstanding shares.",
        "What is a P/E ratio, and how is it used in stock analysis?":"The Price-to-Earnings (P/E) ratio is a measure of a company's valuation. It is calculated by dividing the stock price by the earnings per share. A high P/E ratio may indicate high expectations for future earnings growth.",
        "How does dividend yield impact stock investing?":"Dividend yield is the annual dividend income as a percentage of the stock's current market price. Investors often consider a stock's dividend yield as it provides an indication of the income potential from holding that stock.",
        "Explain the difference between a bull market and a bear market.":"A bull market is characterized by rising stock prices and optimism, while a bear market is characterized by falling stock prices and pessimism. These terms are often used to describe the overall market sentiment.",
        "What is insider trading?":"Insider trading involves buying or selling a security in breach of a fiduciary duty or other relationship of trust and confidence while in possession of material, nonpublic information about the security.",
        "How can I buy stocks?": "To buy stocks, open a brokerage account, fund it, research stocks, and place buy orders through the broker's platform.",
        "What is a stock symbol?": "A stock symbol is a unique series of letters assigned to a publicly traded company to identify its stock on exchanges (e.g., AAPL for Apple Inc).",
        "How do I analyze a stock?": "Stock analysis involves examining a company's financial health, market trends, and other factors. Methods include fundamental and technical analysis.",
        "How to stay updated on stock market news?": "Stay updated by following financial news websites, subscribing to market newsletters, and using financial news apps.",
        # Add more questions and answers as needed
    }
    def answer_stock_question(user_question):
    
        st.header("Stock Knowledge Chatbot")

        question_list = list(stock_questions.keys())
        selected_question = st.radio("Select a question", question_list)
        st.markdown(f"*You:* {selected_question}")

        if selected_question != "Select a question":
            system_response = stock_questions.get(selected_question,"no")
    # Display system response
            st.markdown(f"*Chatbot:* {system_response}")

    answer_stock_question(stock_questions)

st.write("---")
st.subheader('Visualization')

#visualization
#simple closing price chart // chart - 1
#c5
tab1, tab2 , tab3 = st.tabs(['chart1','chart2','chart3'])
with tab1:     
     st.subheader('Close Price Vs Time Chat')
     fig = plt.figure(figsize=(12,6))
     plt.plot(df.Close)
     st.pyplot(fig)

#now its to just add moving avg in it
# so chart -2 with 100MA for that define var and plot that on the chart too with diff color
#c5
with tab2:
    st.subheader('Close Price Vs Time Chat with 100MA') 
    ma100 = df.Close.rolling(100).mean()
    # ma100 = df.Close.rolling(100).mean
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)


# now we need to add 200 ma too so again same chart -3 with 200 ma too
#c5
with tab3:
     st.subheader('Close Price Vs Time Chat with 100MA & 200') 
     ma100 = df.Close.rolling(100).mean()
     ma200 = df.Close.rolling(200).mean()
     fig = plt.figure(figsize=(12,6))
     plt.plot(ma100)
     plt.plot(ma200)
     plt.plot(df.Close)
     st.pyplot(fig)


st.write("---")
st.subheader('predection')

# Splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) #closing price 0 to 70%
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))]) #closing price 70% to full

# for scaling data bw 0 and 1
from sklearn.preprocessing import MinMaxScaler
# scal the data bw 0 and 1 by scalar object using sklearn.preprocessing package
scalar = MinMaxScaler(feature_range=(0,1))

# and convert the data_training into array
data_training_array = scalar.fit_transform(data_training)

# array is divided into x_train = first 100/10 days
# nd y_train = 101/11 index 
# price of 10 days stock and want to predict 11th day price than x_train = 10 days and y_train = 11th day
# x = 34 36 33 40 39 38 37 42 444 38        y = 11th day prediction = 43
# for next, top of x_train(34) removed and first element of y_train will be added to predict 12th day prediction


# tf.compat.v1.get_default_graph

model = load_model('keras_model.h5')


# testing part : get the 100 days data from training
past_100_days = data_training.tail(100)
# apped dtesting
final_df = past_100_days._append(data_testing, ignore_index= True)

#scale testing data
input_data = scalar.fit_transform(final_df)

# print(input_data.shape) # 328,1

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

#convert into numpy array
x_test, y_test = np.array(x_test), np.array(y_test)

# making prediction

y_predicated = model.predict(x_test)

scalar = scalar.scale_
#divided y_test and y_predicated by factor
scale_factor = 1/scalar[0]
y_predicated = y_predicated * scale_factor
y_test = y_test * scale_factor

#plot both value
st.subheader(' Prediction vs Original ')
fig = plt.figure(figsize=(12,6))
plt.plot(y_test,label='Original value',color='blue')
plt.plot(y_predicated,label='predicated value',color='red')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)







