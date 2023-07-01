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
import pyinflect
import spacy
import contractions
import contractions

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
fp = open("/content/Little_Red_Cap_ Jacob_and_Wilhelm_Grimm.txt")
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

model_g = api.load('word2vec-google-news-300')

def opt(row):
    a = set()
    if row['task'] == 'select_word':
        try:
            a.add(row['word'])
            for i in model_g.similar_by_word(row['word'], topn=2):
                a.add(i[0].lower())

            return list(a)
        except:
            return row['word']

df['options'] = df.apply(opt, axis=1)
df['options'] = df.apply(lambda x: [] if x['task'] == 'missing_word' else x['options'], axis=1)

df

def chunk(row):
    try:
        p = []
        a = [] + ['object of preposition', 'direct object']
        if row['task'] == 'phrases':
            for chunk in nlp(row['sentences']).noun_chunks:
                if len(chunk.text) >=7:
                    p.append(chunk.text)
                    a.append(spacy.explain(chunk.root.dep_))
            v = dict(zip(p,a))
            q = np.random.choice(list(p))
        return q, list(set(a)), spacy.explain(chunk.root.dep_)
    except:
        pass

print(['give'] + ['priv'])

df['answer'] = 0

df

ddf = df.copy()

ddf

for i in range(len(ddf)):
    if ddf.loc[i, 'task'] == 'phrases':
        try:
            q = chunk(ddf.loc[i])
            ddf.loc[i,'word'] = list(q)[0]
            ddf.at[i,'options'] = list(q)[1]
            ddf.loc[i,'answer'] = list(q)[2]
        except:
            pass

for i in range(len(ddf)):
    if (ddf.loc[i, 'task'] == 'select_word' or ddf.loc[i, 'task'] == 'missing_word'):
        ddf.loc[i,'answer'] = ddf.loc[i, 'word']

ddf

def sentgen(row):
    if row['task'] == 'select_sent':
        try:
            d = [token for token in nlp(row['sentences']) if token.pos_ in ['VERB']]
            x = np.random.choice(d)
            t = [str(x), x._.inflect('VBP'), x._.inflect('VBG'), x._.inflect('VBD')]
            s = []
            if t[0] in (row['sentences'].split(' ')):
                sent_1 = row['sentences'].replace(t[0], t[1])
                s.append(sent_1)
                sent_2 = row['sentences'].replace(t[0], t[2])
                s.append(sent_2)
                sent_3 = row['sentences'].replace(t[0], t[3])
                s.append(sent_3)
            return row['sentences'], s
        except:
            pass

for i in range(len(ddf)):
    if ddf.loc[i, 'task'] == 'select_sent':
        try:
            s = sentgen(ddf.loc[i])
            ddf.loc[i,'word'] = list(s)[0]
            ddf.at[i,'options'] = list(s)[1]
            ddf.loc[i,'answer'] = list(s)[0]
        except:
            pass

ddf


import streamlit as st

def main():
    st.title('Генератор упражнений по английскому языку.')

if __name__ == '__main__':
    main()


