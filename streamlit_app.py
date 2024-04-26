import numpy as np
import pandas as pd
import re # regular expression for pattern matching
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # reduce word to it's root word
import streamlit as st 
from textblob import TextBlob
import pickle
import joblib

st.set_page_config(layout='wide')

st.header('TWITTER SENTIMENT ANALYSIS WEB APPLICATION')

st.subheader('Welcome to my Twitter Sentiment analysis Web Application.')

st.write('Harness the power of sentiment analysis with this application, your go-to tool for understanding public opinion on Twitter. Our application utilizes advanced natural language processing techniques to analyze tweets in real-time and provide valuable insights into sentiment trends.')

st.write('Whether you\'re tracking brand perception, monitoring public opinion, or simply curious about trending topics, SentimentScope is your all-in-one solution.')

st.subheader('To get started...')
st.write('Select any one of the two options listed below.')

with st.expander('ANALYSE TWEET'):
    user_input = st.text_input('Enter your text below','I am feeling awesome today!')

    # processing user input
    nltk.download('stopwords')

    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]',' ',user_input)#remove everything that isn't a letter
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)

    print(stemmed_content)
    test = []
    test.append(stemmed_content)

    vectorizer = joblib.load('vectorizer.pkl')
    test = vectorizer.transform(test)

    model = pickle.load(open('trained_model.sav','rb'))

    prediction = model.predict(test)

    if(prediction == 0):
        result = "Oh! It's a NEGATIVE tweet... :("
    else:
        result = "Hurray!! It's POSITIVE tweet... :)"

    if st.button("Analyze the Sentiment"): 
        st.write(result)

with st.expander('ANALYSE EXCEL SHEET'):
    data_xl = st.file_uploader('Upload file')
    st.write('Note: The name of the tweets column should be "tweets".')
    
    def analyse(x):
        if(x == 0):
            return "NEGATIVE"
        else:
            return "POSITIVE"
    
    if st.button('Analyse the Sentiment') and data_xl:
        df = pd.read_excel(data_xl)
        def stemming(content):
            stemmed_content = re.sub('[^a-zA-Z]',' ',content)#remove everything that isn't a letter
            stemmed_content = stemmed_content.lower()
            stemmed_content = stemmed_content.split()
            stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
            stemmed_content = ' '.join(stemmed_content)
            return stemmed_content
        
        data = df['tweets'].apply(stemming)
        data = data.values
        data = vectorizer.transform(data.astype(str))
        df['sentiment'] = (model.predict(data))
        df['sentiment'].replace({1:'Positive', 0:'Negative'}, inplace = True)
        st.write(df)
