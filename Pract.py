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

def gen_ex(text, num):

    my_bar = st.progress(0, text='Wait')
    splitter = SentenceSplitter(language='en')
    sentences = splitter.split(text=text)
    my_bar.progress(20, text='Разбивка текста')
    df = pd.DataFrame()

    l = []
    for i in sentences:
        if i == "":
            pass
        else:
            l.append(i)


    df['sentences'] = l

    df['sentences'] = df['sentences'].apply(lambda x: contractions.fix(x))

    df.loc[:, 'task'] = df.apply(lambda x: np.nan if len(x['sentences'].split()) <= 7 else np.random.choice(
        ['select_word', 'missing_word', 'phrases', 'select_sent']), axis=1)

    nlp = en_core_web_sm.load()
    my_bar.progress(40, text='Загрузка модели')
    
    def obj(row):
        z = []
        if (row['task'] == 'select_word' or row['task'] =='missing_word'):
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

    model_g = api.load('glove-wiki-gigaword-100')
    my_bar.progress(60, text='Загрузка еще одной модели')

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

    df['answer'] = 0
    my_bar.progress(80, text='Осталось чуть-чуть')

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

    for i in range(len(df)):
        if df.loc[i, 'task'] == 'phrases':
            try:
                q = chunk(df.loc[i])
                df.loc[i,'word'] = list(q)[0]
                df.at[i,'options'] = list(q)[1]
                df.loc[i,'answer'] = list(q)[2]
            except:
                pass

    for i in range(len(df)):
        if (df.loc[i, 'task'] == 'select_word' or df.loc[i, 'task'] == 'missing_word'):
            df.loc[i,'answer'] = df.loc[i, 'word']

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

    for i in range(len(df)):
        if df.loc[i, 'task'] == 'select_sent':
            try:
                s = sentgen(df.loc[i])
                df.loc[i,'word'] = list(s)[0]
                df.at[i,'options'] = list(s)[1]
                df.loc[i,'answer'] = list(s)[0]
            except:
                pass
    df = df.dropna()
    df = df.sample(num)
    my_bar.progress(100, text='Готово')
    st.write('Генерация завершена')
    return df

def main(text, num):

    df = pd.DataFrame()  # Инициализация пустого DataFrame

    if st.button("Сгенерировать"):
        df = gen_ex(text, num)  # Сохранение сгенерированного DataFrame

    if not df.empty:  # Проверка наличия DataFrame перед использованием
        form = st.form(key='exercise_form')

        if form.form_submit_button("Проверить все ответы"):
            pass

        for i, row in df.iterrows():
            with form:
                sentence = row['sentences']
                odj = row['word']
                task = row['task']
                option = row['options']
                answ = row['answer']
    
                st.subheader(f'{i+1} упражнение')
    
                if task == 'select_word':
                    st.write(sentence)
                elif task == 'missing_word':
                    words = sentence.split()
                    if answ in words:
                        ind = words.index(answ)
                        words[ind] = '_' * len(words[ind])
                        missing_word_sentence = ' '.join(words)
                        st.write(missing_word_sentence)
        
                        user_answer = st.text_input('Введите ваш ответ:', key=f'{i}')
                        check_button = form.form_submit_button(f'Проверить {i}')
        
                        if check_button:
                            if user_answer.lower() == answ.lower():
                                st.success('Верно!', icon="✅")
                            else:
                                st.error('Ошибка', icon="🚨")
                            st.write(sentence)
                elif task == 'phrases':
                    st.write(sentence)
                    option_str = ', '.join(option)
                    st.write(f'Варианты ответов: {option_str}')
        
                    st.write(answ)
                    user_answer = st.selectbox(f'Выберите правильный ответ {i}:', ['', *option])
                    check_button = form.form_submit_button(f'Проверить {i}')
        
                    if check_button:
                        if user_answer.lower() == answ.lower():
                            st.success('Правильный ответ!')
                        else:
                            st.error('Неправильный ответ!')
                elif task == 'select_sent':
                    st.write(sentence)
                    st.write('Варианты предложений:')
                    for j, opt in enumerate(option):
                        st.write(f'{j + 1}. {opt}')
        
                    user_answer = st.selectbox(f'Выберите правильное предложение {i}:', ['', *option])
                    check_button = form.form_submit_button(f'Проверить {i}')
        
                    if check_button:
                        if user_answer.lower() == answ.lower():
                            st.success('Правильный ответ!')
                        else:
                            st.error('Неправильный ответ!')
            


if __name__ == '__main__':
    
    st.title("Генератор упражнений по английскому")
    text = st.text_area("Введите текст:", key="text_area")
    num = st.number_input("Введите количество упражнений:", min_value=1, step=1, key='num')
    main(text, num)
