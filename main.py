import streamlit as st
import requests
import nltk
import pandas as pd
from io import StringIO
import readtime
import time
import textstat
from nltk import sent_tokenize, word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lex_rank import LexRankSummarizer
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Navbar
selected = option_menu (
    menu_title = None,
    options = ['URL Analysis', 'Text File Analysis', 'Dataset Analysis'],
    icons = ['link-45deg', 'file-text', 'bar-chart-line'],  
    menu_icon = 'cast',
    default_index = 0,
    orientation = 'horizontal',
)

st.title("Welcome to EcoRead")

# Introducing the user how to use the application
st.markdown('___')
st.write(':point_left: Use the sidebar on the left to learn about EcoRead (click on > if closed).')

st.divider()

# SIDEBAR
st.sidebar.markdown("<h3 style='text-align: left; color:#F63366; font-size:18px;'><b>What is this App about?<b></h3>", unsafe_allow_html=True)
st.sidebar.info(
    """
    Learning happens best when content is personalised to meet our needs and strengths.     
    """
)
st.sidebar.write("---")
 
st.sidebar.markdown(
    """
    EcoRead allows you to analyse the sentiment of an article by providing a URL, or by uploading a text file or by using the pre-existing dataset about climate change.
    """
)
st.sidebar.divider()

st.sidebar.write("Are you into NLP? Our code is 100% open source and written for easy understanding. Fork it from [GitHub](https://github.com/akebu6/EcoRead), and pull any suggestions you may have. Become part of the community! Help yourself and help others :smiley:")

st.sidebar.divider()

st.sidebar.markdown("<h3 style='text-align: left; color:#F63366; font-size:18px;'><b>Who is this App for?<b></h3>", unsafe_allow_html=True)
st.sidebar.write("Anyone can use this App completely for free! If you like it :heart:, show your support by sharing :+1: ")

st.sidebar.divider()

#CONTACT
########
expander = st.sidebar.expander('Contact')
expander.write(
    "I'd love your feedback :smiley: Want to collaborate? Develop a project? Find me on [LinkedIn](https://www.linkedin.com/in/akebu-simasiku-24186720a/) and [Twitter](https://twitter.com/akebu)"
)

# Function to extract text from a URL
def extract_text_from_url(url):
    response = requests.get(url)
    return response.text

# Function to tokenize text and identify keywords
def identify_keywords(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    words = [word for word in words if word not in stopwords.words('english')]
    freq_dist = FreqDist(words)
    return freq_dist.most_common(10)  # Return top 10 keywords

def summarize_text(text):
    sentences = sent_tokenize(text)
    return ' '.join(sentences[:3])  # Return the first 3 sentences as a summary

# Function to analyze sentiment
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    sentiment = 'positive' if sentiment_scores['compound'] > 0 else 'negative' if sentiment_scores['compound'] < 0 else 'neutral'
    return sentiment


# User Interaction
if selected == "URL Analysis":
    url = st.text_input("Enter the URL of the climate change article and press Enter or Return: 👇 ",  placeholder="Enter a valid URL")

    if st.button('Search'):
        text = extract_text_from_url(url)
        keywords = identify_keywords(text)
        summary = summarize_text(text)
        sentiment = analyze_sentiment(text)
        st.write("\nKeywords:", [keyword for keyword, _ in keywords])
        st.write("Summary:", summary)
        st.write("Sentiment:", sentiment)
        
        # graph visualization
        # Graph Visualization
        plt.bar(*zip(*keywords))
        plt.xlabel('Keywords')
        plt.ylabel('Frequency')
        plt.title('Top Keywords in the Article')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Uploading a text file
if selected == 'Text File Analysis':
        file = st.file_uploader('Upload your file here',type=['txt'])
        if file is not None:
            with st.spinner('Processing...'):
                    time.sleep(2)
                    stringio = StringIO(file.getvalue().decode("utf-8"))
                    string_data = stringio.read()
                    if len(string_data) > 10000:
                        st.error('Please upload a file of maximum 10,000 characters')
                    else:
                        nltk.download('punkt')
                        rt = readtime.of_text(string_data)
                        tc = textstat.flesch_reading_ease(string_data)
                        tokenized_words = word_tokenize(string_data)
                        lr = len(set(tokenized_words)) / len(tokenized_words)
                        lr = round(lr,2)
                        n_s = textstat.sentence_count(string_data)
                        st.markdown('___')
                        st.text('Reading Time')
                        st.write(rt)
                        st.markdown('___')
                        st.text('Text Complexity: from 0 or negative (hard to read), to 100 or more (easy to read)')
                        st.write(tc)
                        st.markdown('___')
                        st.text('Lexical Richness (distinct words over total number of words)')
                        st.write(lr)
                        st.markdown('___')
                        st.text('Number of sentences')
                        st.write(n_s)
                        st.balloons()
                        
# Dataset Analysis
if selected == "Dataset Analysis":
    df = pd.read_csv()
    