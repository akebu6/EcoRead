import streamlit as st
import nltk
import requests
import pandas as pd
from bs4 import BeautifulSoup

import time
import readtime
import textstat
from io import StringIO
import heapq

from nltk import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

import plotly.express as px
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
    EcoRead allows you to analyse the sentiment of a pre-existing dataset, summarise the contents of an article for quicker information digestion as we well as analyse the complexity of a text.
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

# Function to extract text content from a URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')  # Adjust this based on the HTML structure
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

# Function to generate a summarized version of the text
def generate_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)
    words = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    word_freq = FreqDist(words)
    ranking = {}
    
    for i, sentence in enumerate(sentences):
        for word, freq in word_freq.items():
            if word in sentence.lower():
                if i not in ranking:
                    ranking[i] = freq
                else:
                    ranking[i] += freq
                    
    selected_sentences = heapq.nlargest(num_sentences, ranking, key=ranking.get)
    summary = [sentences[i] for i in sorted(selected_sentences)]
    return ' '.join(summary)


# User Interaction
if selected == "URL Analysis":
    url = st.text_input("Enter the URL of the climate change article and press Enter or Return: 👇 ",  placeholder="Enter a valid URL")

    if st.button('Search'):
        with st.spinner('Processing...'):
            time.sleep(2)
            text = extract_text_from_url(url)
            summary = generate_summary(text)
            
            st.write("\nSummary:")
            st.write(summary)


# Uploading a text file
if selected == 'Text Complexity Analysis':
    file = st.file_uploader('Upload your file here', type=['txt'])
    if file is not None:    
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
            # st.balloons()   
             
                        
# Dataset Analysiso plot the data
if selected == "Dataset Analysis":
    PATH = "./csv_files/"
    DATA = "twitter_sentiment_data.csv"
    data = pd.read_csv(PATH + DATA)
    
    # set up label dataframe for future refrences

    label = [-1, 0, 1, 2]
    labelN = ["Anti", "Neutral", "Pro", "News"]
    labelDesc = [
        "the tweet does not believe in man-made climate change"
        , "the tweet neither supports nor refutes the belief of man-made climate change"
        , "the tweet supports the belief of man-made climate change"
        , "the tweet links to factual news about climate change"
    ]

    labelDf = pd.DataFrame(list(zip(label, labelN, labelDesc)), columns=["label", "name", "description"])
    
    fig = px.pie(data.sentiment.value_counts().values, 
                 data.sentiment.value_counts().index, title='Sentiment Distribution of the Tweet Dataset')

    
    st.plotly_chart(fig)
     