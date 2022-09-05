import streamlit as st
import plotly.graph_objs as go
import yfinance as yf
import os
import collections
from wordcloud import WordCloud

import tweepy
from datetime import datetime, timedelta
import pickle
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import re
import nltk
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from nltk.stem.wordnet import WordNetLemmatizer
from html.parser import HTMLParser
from nltk.tokenize import word_tokenize

st.set_page_config(page_title="Dashboard", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle

LemmatizerInstance = WordNetLemmatizer()
HTMLParserInstance = HTMLParser()

f = open("dict_apostrophe.pickle", "rb")
apostrophe_dict = pickle.load(f)
f = open("dict_short.pickle", "rb")
short_word_dict = pickle.load(f)
f = open("dict_emoji.pickle", "rb")
emoticon_dict = pickle.load(f)

with open("model.pickle", "rb") as f:
    svc_clf = pickle.load(f)
f.close()

with open("tfidf_vectorizer.pickle", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
f.close()

def get_actual_prices(crypto_type, color):
    ticker = yf.Ticker(f"{crypto_type}-USD")
    data = ticker.history(period="6d", interval="1m")
    fig = go.Figure(data=go.Scatter(x=data["Open"].index, 
                            y=data["Open"].values,
                            marker_color=color, text="Price(USD)"))
    fig.update_layout({"title": f'Actual {crypto_type} Prices from {str(min(data.index)).split(" ")[0]} to {str(max(data.index)).split(" ")[0]}',
                    "xaxis": {"title":"Date"},
                    "yaxis": {"title":"Price(USD)"},
                    "showlegend": False})
    return fig

def FunctionDict(t, d):
    '''This function splits the text into words and replace the word with the value mapped in the dictionary if present'''
    for w in t.split():
        if w.lower() in d:
            if w.lower() in t.split():
                t = t.replace(w, d[w.lower()])
    return t

def get_clean_text(text):
    cleaned_text = text.replace("\n", " ")
    cleaned_text = HTMLParserInstance.unescape(cleaned_text)
    cleaned_text = cleaned_text.lower()
    cleaned_text = FunctionDict(cleaned_text, apostrophe_dict)
    cleaned_text = FunctionDict(cleaned_text, short_word_dict)
    cleaned_text = FunctionDict(cleaned_text, emoticon_dict)
    cleaned_text = re.sub(r'[^\w\s]',' ', cleaned_text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9]',' ', cleaned_text)
    cleaned_text = re.sub(r'[^a-zA-Z]',' ', cleaned_text)
    cleaned_text = ' '.join([w for w in cleaned_text.split() if len(w)>1])
    cleaned_text = word_tokenize(cleaned_text)
    cleaned_text = [w for w in cleaned_text if not w in stop_words]
    cleaned_text = ' '.join([LemmatizerInstance.lemmatize(i) for i in cleaned_text])
    return cleaned_text

def scrap_load_data():
    bearer_token = 'AAAAAAAAAAAAAAAAAAAAAIF%2FfQEAAAAAlRsrX61Bg3Bho%2Fv0n0JW4Ufa8rA%3Dr5WfagCULkXtF8KnVRksOsmp2wM2w6StO1e4XLqNiJ9QlEV7RK'
    client = tweepy.Client(bearer_token=bearer_token)
    all_tweets_dict = {}
    all_counts_dict = {}
    for i in range(0, 6):
        start_time = (datetime.now() - timedelta(days=1, hours=6) - timedelta(days=i)).strftime("%Y-%m-%dT%H:%M:%S%ZZ")
        end_time = (datetime.now() - timedelta(hours=6) - timedelta(days=i)).strftime("%Y-%m-%dT%H:%M:%S%ZZ")
        queries = ['#Ethereum -is:retweet lang:en', '#Litecoin -is:retweet lang:en', '#Bitcoin -is:retweet lang:en']
        day_tweets = []
        day_counts = []
        all_tweets_dict[end_time.split("T")[0]] = day_tweets
        all_counts_dict[end_time.split("T")[0]] = day_counts
        for query in queries:
            for tweet in tweepy.Paginator(client.search_recent_tweets, query=query, start_time=start_time, end_time=end_time, max_results=100).flatten(limit=50):
                day_tweets.append(get_clean_text(tweet.text))
            for counts in client.get_recent_tweets_count(query=query, start_time=start_time, end_time=end_time):
                if (type(counts) == dict ) & (len(counts) != 0):
                    day_counts.append(counts.get("total_tweet_count"))
    all_counts_dict = dict(zip(all_counts_dict.keys(), [sum(i) for i in all_counts_dict.values()]))
    return all_tweets_dict, all_counts_dict


def load_tweets_info():
    with open("all_tweets_dict.pkl", "rb") as f:
        all_tweets_dict = pickle.load(f)
    f.close()
    with open("all_counts_dict.pkl", "rb") as f:
        all_counts_dict = pickle.load(f)
    f.close()
    return all_tweets_dict, all_counts_dict



def get_pred_dict(all_tweets_dict):
    prediction_dict = {}
    positive_ratio_dict = {}
    for day in all_tweets_dict.keys():
        tweets = all_tweets_dict[day]
        tfidf_tweets = tfidf_vectorizer.transform(tweets)
        predictions = svc_clf.predict(tfidf_tweets)
        prediction_dict[day] = predictions
        positive_ratio_dict[day] = np.count_nonzero(predictions) / len(predictions)
    return prediction_dict, positive_ratio_dict



def plot_pos_sent(x, y):
    fig = go.Figure(data=go.Scatter(x=x, 
                            y=y,
                            marker_color='indianred', text="Ratio"))
    fig.update_layout({"title": f'Positive Sentiment Ratio from {min(x)} to {max(x)}',
                    "xaxis": {"title":"Date"},
                    "yaxis": {"title":"Positive Sentiment Ratio"},
                    "showlegend": False})
    return fig



def plot_tweet_count(x, y):
    fig = go.Figure(data=go.Scatter(x=x, 
                            y=y,
                            marker_color='violet', text="Counts"))
    fig.update_layout({"title": f'Crypto Tweet Counts from {min(x)} to {max(x)}',
                    "xaxis": {"title":"Date"},
                    "yaxis": {"title":"Total Tweet Counts"},
                    "showlegend": False})
    return fig

def get_donut(data):
    colors = ['limegreen', '#800080']
    labels = ["Negative", "Positive"]
    explode = (0.10, 0)
    fig, ax = plt.subplots()
    fig.set_facecolor("#fff9c9")
    plt.pie(data, labels=labels, colors=colors, explode=explode, autopct="%1.1f%%")
    centre_circle = plt.Circle((0, 0), 0.60, fc='#fff9c9')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    return fig


def get_wordcloud(text_list):
    WordString = ' '.join(text_list)
    wordcloud = WordCloud(background_color="white").generate(WordString)
    fig = plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return fig


def display_donuts():
    if os.path.exists("all_tweets_dict.pkl"):

        all_tweets_dict, _ = load_tweets_info()
        predictions, _ = get_pred_dict(all_tweets_dict)
        pie_data = {}
        for day in predictions.keys():
            pie_data[day] = (list(predictions[day]).count(0), list(predictions[day]).count(1))
        pie_data = collections.OrderedDict(sorted(pie_data.items()))

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        with col1:
            st.header(list(pie_data.keys())[0])
            fig = get_donut(pie_data[list(pie_data.keys())[0]])
            st.pyplot(fig)

        with col2:
            st.header(list(pie_data.keys())[1])
            fig = get_donut(pie_data[list(pie_data.keys())[1]])
            st.pyplot(fig)

        with col3:
            st.header(list(pie_data.keys())[2])
            fig = get_donut(pie_data[list(pie_data.keys())[2]])
            st.pyplot(fig)

        with col4:
            st.header(list(pie_data.keys())[3])
            fig = get_donut(pie_data[list(pie_data.keys())[3]])
            st.pyplot(fig)

        with col5:
            st.header(list(pie_data.keys())[4])
            fig = get_donut(pie_data[list(pie_data.keys())[4]])
            st.pyplot(fig)
        with col6:
            st.header(list(pie_data.keys())[5])
            fig = get_donut(pie_data[list(pie_data.keys())[5]])
            st.pyplot(fig)
    else:
        st.error("Please scrap the data first!")

def display_wordclouds():
    if os.path.exists("all_tweets_dict.pkl"):

        all_tweets_dict, _ = load_tweets_info()

        for day in all_tweets_dict.keys():
            text_list_clean = []
            for text in all_tweets_dict[day]:
                text = text.replace(" co ", " ")
                text_list_clean.append(text)
            all_tweets_dict[day] = text_list_clean


        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        with col1:
            st.header(str(sorted(list(all_tweets_dict.keys()))[0]))
            fig = get_wordcloud(all_tweets_dict[str(sorted(list(all_tweets_dict.keys()))[0])])
            st.pyplot(fig)

        with col2:
            st.header(str(sorted(list(all_tweets_dict.keys()))[1]))
            fig = get_wordcloud(all_tweets_dict[str(sorted(list(all_tweets_dict.keys()))[1])])
            st.pyplot(fig)

        with col3:
            st.header(str(sorted(list(all_tweets_dict.keys()))[2]))
            fig = get_wordcloud(all_tweets_dict[str(sorted(list(all_tweets_dict.keys()))[2])])
            st.pyplot(fig)

        with col4:
            st.header(str(sorted(list(all_tweets_dict.keys()))[3]))
            fig = get_wordcloud(all_tweets_dict[str(sorted(list(all_tweets_dict.keys()))[3])])
            st.pyplot(fig)

        with col5:
            st.header(str(sorted(list(all_tweets_dict.keys()))[4]))
            fig = get_wordcloud(all_tweets_dict[str(sorted(list(all_tweets_dict.keys()))[4])])
            st.pyplot(fig)

        with col6:
            st.header(str(sorted(list(all_tweets_dict.keys()))[5]))
            fig = get_wordcloud(all_tweets_dict[str(sorted(list(all_tweets_dict.keys()))[5])])
            st.pyplot(fig)
    else:
        st.error("Please scrap the data first!")


with st.sidebar:
    title = "Dashboard"
    st.title(title)
    st.write("Welcome to the Crypto Sentiment Analysis Dashboard!")



actual_prices_bar = st.sidebar.radio("Get Actual Crypto Prices:", ("Bitcoin", "Ethereum", "Litecoin"))
if actual_prices_bar == "Bitcoin":
    st.plotly_chart(get_actual_prices("BTC", "indianred"), use_container_width=True)
elif actual_prices_bar == "Ethereum":
    st.plotly_chart(get_actual_prices("ETH", "green"), use_container_width=True)
elif actual_prices_bar == "Litecoin":
    st.plotly_chart(get_actual_prices("LTC", "orange"), use_container_width=True)
    

scrap_data_bar = st.sidebar.button("Scrap Latest Twitter Data")
if scrap_data_bar == True:
    with st.spinner("Scraping data...... (ETA: 10 Seconds)"):
        all_tweets_dict, all_counts_dict = scrap_load_data()
        with open("all_tweets_dict.pkl", "wb") as f:
            pickle.dump(all_tweets_dict, f)
        f.close()
        with open("all_counts_dict.pkl", "wb") as f:
            pickle.dump(all_counts_dict, f)
        f.close()
        # time.sleep(2)
        st.sidebar.success("Successfully scraped. You may use all functions now!")

plot_scrap_bar = st.sidebar.radio("Plot:", ("Positive Sentiment Ratio", "Crypto Tweet Count"))
if plot_scrap_bar == "Positive Sentiment Ratio":
    if os.path.exists("all_tweets_dict.pkl"):
        all_tweets_dict, _ = load_tweets_info()
        _, pos_ratio_dict = get_pred_dict(all_tweets_dict)
        x, y = zip(*sorted(pos_ratio_dict.items()))
        fig = plot_pos_sent(x, y)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.sidebar.info("Please scrap the data first!")
if plot_scrap_bar == "Crypto Tweet Count":
    if os.path.exists("all_tweets_dict.pkl"):
        _, all_counts_dict = load_tweets_info()
        x, y = zip(*sorted(all_counts_dict.items()))
        fig = plot_tweet_count(x,y)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.sidebar.info("Please scrap the data first!")



wc_bar = st.sidebar.button("Display Wordclouds", on_click=display_wordclouds)
if wc_bar == True:
    if os.path.exists("all_tweets_dict.pkl"):
        st.sidebar.success("Fetched WordClouds!")
    else:
        st.sidebar.error("Need scraped data!!")

donut_bar = st.sidebar.button("Display Pie Donuts", on_click=display_donuts)
if donut_bar == True:
    if os.path.exists("all_tweets_dict.pkl"):
        st.sidebar.success("Fetched Pie Donuts!")
    else:
        st.sidebar.error("Need scraped data!!")


txt_bar = st.sidebar.text_area('Enter Text to predict:', placeholder="Bitcoin is the best crypto...")
if txt_bar != "":
    tfidf_cvt = tfidf_vectorizer.transform([txt_bar])
    pred = svc_clf.predict(tfidf_cvt)
    if pred[0] == 1:
        sentiment = "Positive"
    else:
        sentiment = "Negative"
    st.sidebar.info(f"Sentiment: {sentiment}")



