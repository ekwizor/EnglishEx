# -*- coding: utf-8 -*-
import nltk
import pandas as pd
import nltk.data
import seaborn as sns
from sentence_splitter import SentenceSplitter
nltk.download('stopwords')
nltk.download('punkt')
import gensim.downloader as api
import en_core_web_sm
import numpy as np
from nltk.corpus import stopwords
import pyinflect
import spacy
import contractions
import streamlit as st

st.title('Генератор упражнений по английскому языку.')

def make_dataframe():
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    fp = open("Little_Red_Cap_ Jacob_and_Wilhelm_Grimm.txt")
    data = fp.read()
    
    splitter = SentenceSplitter(language='en')
    sentences = splitter.split(text=data)
    
    df = pd.DataFrame()
    
    l = []
    for i in sentences:
        if i == "":
            pass
        else:
            l.append(i)
    df['sentences'] = l
    return df

df = make_dataframe()

def sentfix(row):
    return contractions.fix(row)

df['sentences'] = df['sentences'].apply(sentfix)





st.title('Генерация готова')





