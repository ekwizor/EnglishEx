import pandas as pd
import numpy as np
import en_core_web_sm
import streamlit as st
from sentence_splitter import SentenceSplitter
import gensim.downloader as api
import contractions
import string
import spacy
def split_sentences(text):
  splitter = SentenceSplitter(language='en')
  sentences = splitter.split(text=text)
  sentences = [sentence for sentence in sentences if sentence != ""]
  return sentences
def generate_df(sentences):
  df = pd.DataFrame()
  df['sentences'] = sentences
  df['sentences'] = df['sentences'].apply(lambda x: contractions.fix(x))
  df.loc[:, 'task'] = df.apply(lambda x: np.nan if len(x['sentences'].split()) <= 7 else np.random.choice(
      ['select_word', 'missing_word', 'phrases', 'select_sent']), axis=1)
  return df
model_g = api.load('glove-wiki-gigaword-100')  # Load the model only once
def process_df(df, nlp):
  # Processing 'select_word' and 'missing_word' tasks
  df.loc[df['task'].isin(['select_word', 'missing_word']), 'word'] = df.apply(lambda row: process_word(row, nlp), axis=1)
   # Processing 'select_word' tasks
  df.loc[df['task'] == 'select_word', 'options'] = df.apply(lambda row: process_options(row, model_g), axis=1)
   # Processing 'phrases' tasks
  df.loc[df['task'] == 'phrases', ['word', 'options', 'answer']] = df.apply(lambda row: process_phrases(row, nlp), axis=1)
   # Processing 'select_word' and 'missing_word' tasks
  df.loc[df['task'].isin(['select_word', 'missing_word']), 'answer'] = df['word']
   # Processing 'select_sent' tasks
  df.loc[df['task'] == 'select_sent', ['word', 'options', 'answer']] = df.apply(lambda row: process_select_sent(row, nlp), axis=1)
  df = df.dropna()
  return df
def process_word(row, nlp):
  z = (str(token) for token in nlp(row['sentences']) if token.pos_ in ['VERB', 'ADJ'])  # Use a generator instead of a list
  return np.random.choice(z) if z else None
def process_options(row, model_g):
  a = set()
  try:
      a.add(row['word'])
      for i in model_g.similar_by_word(row['word'], topn=2):
          a.add(i[0].lower())
      return list(a)
  except KeyError:  # Specify the type of exception
      return row['word']
def process_phrases(row, nlp):
  try:
      p = []
      a = set(['object of preposition', 'direct object'])
      for chunk in nlp(row['sentences']).noun_chunks:
          if len(chunk.text) >= 7:
              p.append(chunk.text)
              a.add(spacy.explain(chunk.root.dep_))
      v = dict(zip(p, list(a)))
      q = np.random.choice(list(p))
      return q, list(a), spacy.explain(chunk.root.dep_)
  except ValueError:  # Specify the type of exception
      return None, None, None
def process_select_sent(row, nlp):
  try:
      d = [str(token) for token in nlp(row['sentences']) if token.pos_ in ['VERB']][:2]
      x = np.random.choice(d)
      t = [x, x._.inflect('VBP'), x._.inflect('VBG'), x._.inflect('VBD')]
      s = set([row['sentences']])
      for inflection in t[1:]:
          if t[0] in row['sentences']:
              s.add(row['sentences'].replace(t[0], inflection))
      return row['sentences'], list(s), row['sentences']
  except ValueError:  # Specify the type of exception
      return None, None, None
def main(text, num, nlp):
  
  if 'df' not in st.session_state:
      st.session_state.df = pd.DataFrame()
  if st.button("Сгенерировать"):
      sentences = split_sentences(text)
      df = generate_df(sentences)
      st.session_state.df = process_df(df, nlp)
  df = st.session_state.df
  df = df.reset_index()
   # The rest of your main function goes here...
if __name__ == '__main__':
  nlp = en_core_web_sm.load()
  st.title("Генератор упражнений по английскому")
  text = st.text_area("Введите текст:", key="text_area", height=300)
  num = st.number_input("Введите количество упражнений:", min_value=1, step=1, key='num')
  main(text, num, nlp)
