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

df.loc[:,'task'] = df.apply(lambda x: np.nan if len(x['sentences'].split())<=7 else np.random.choice(['select_word', 'missing_word', 'phrases', 'select_sent']), axis=1)

nlp = en_core_web_sm.load()

def obj(row):
    z = []
    if (row['task'] == 'select_word' or  row['task'] =='missing_word'):
        for token in nlp(row['sentences']):
            if token.pos_ in ['VERB', 'ADJ'] and len(token) > 2:
                z.append(str(token).lower())
        try:
            return np.random.choice(z)
        except:
            pass
    else:
        pass

df['word'] = df.apply(obj, axis=1)









st.title('Генерация готова')





