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
import string


def gen_ex(text, num, nlp):

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

    #nlp = en_core_web_sm.load()
    my_bar.progress(40, text='Загрузка модели')
    
    def obj(row):
        z = []
        if (row['task'] == 'select_word' or row['task'] =='missing_word'):
            for token in nlp(row['sentences']):
                if token.pos_ in ['VERB', 'ADJ']:
                    z.append(str(token))
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
            a = set() 
            a.update(['object of preposition', 'direct object'])
            if row['task'] == 'phrases':
                for chunk in nlp(row['sentences']).noun_chunks:
                    if len(chunk.text) >=7:
                        p.append(chunk.text)
                        a.add(spacy.explain(chunk.root.dep_))
                v = dict(zip(p,list(a)))
                q = np.random.choice(list(p))
            return q, list(a), spacy.explain(chunk.root.dep_)
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
                d = [token for token in nlp(row['sentences']) if token.pos_ in ['VERB']][:2]
                x = np.random.choice(d)
                t = [str(x), x._.inflect('VBP'), x._.inflect('VBG'), x._.inflect('VBD')]
                s = set()
                s.add(row['sentences'])
                if t[0] in row['sentences']:
                    sent_1 = row['sentences'].replace(t[0], t[1])
                    s.add(sent_1)
                    sent_2 = row['sentences'].replace(t[0], t[2])
                    s.add(sent_2)
                    sent_3 = row['sentences'].replace(t[0], t[3])
                    s.add(sent_3)
                return row['sentences'], list(s)
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

def main(text, num, nlp):
    
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()

    if st.button("Сгенерировать"):
        st.session_state.df = gen_ex(text, num, nlp)
    
    df = st.session_state.df
    df = df.reset_index()


    def remove_punctuation(input_string):
        # Создаем таблицу перевода для удаления знаков препинания
        translator = str.maketrans('', '', string.punctuation)
        
        # Применяем таблицу перевода к строке
        no_punct = input_string.translate(translator)
        
        return no_punct
    

    if not df.empty:
        st.dataframe(df)
        for i, row in df.iterrows():
            st.subheader(f'{i+1} упражнение')

            sentence = row['sentences']
            obj = row['word']
            task = row['task']
            option = row['options']
            answ = row['answer']
            #option.extend(answ)
            
            if task == 'select_word':
                #st.write(sentence)
                words = ' '.join([token.text_with_ws for token in nlp(sentence)]).split()
                if answ in words:
                    ind = words.index(answ)
                    words[ind] = '_' * len(words[ind])
                    missing_word_sentence = ' '.join(words)
                    st.write(missing_word_sentence)
                user_answer = st.selectbox(f'Выберите правильное слово:', [*option], key=f'word_{i}')
                check_button = st.button(f'Проверить', key=f'bword{i}')
    
                if check_button:
                    if user_answer.lower() == answ.lower():
                        st.success('Правильный ответ!')
                    else:
                        st.error('Неправильный ответ!')
            elif task == 'missing_word':
                words = ' '.join([token.text_with_ws for token in nlp(sentence)]).split()
                if answ in words:
                    ind = words.index(answ)
                    words[ind] = '_' * len(words[ind])
                    missing_word_sentence = ' '.join(words)
                    st.write(missing_word_sentence)
                    st.write(f'First letter: {answ[0]}')
                    st.write(f'Last letter: {answ[-1]}')
    
                    user_answer = st.text_input('Введите ваш ответ:', key=f'text_inp_{i}')
                    check_button = st.button(f'Проверить', key=f'bmword{i}')
    
                    if check_button:
                        if user_answer.lower() == answ.lower():
                            st.success('Верно!', icon="✅")
                        else:
                            st.error('Ошибка', icon="🚨")
                        st.write(sentence)
            elif task == 'phrases':
                highlighted_sentence = sentence.replace(obj, f'<span style="color:red">{obj}</span>')
                st.markdown(highlighted_sentence, unsafe_allow_html=True)
                
                st.write('<b>Чем является выделенный фрагмент ?<b>', unsafe_allow_html=True)
                st.write(answ)
                user_answer = st.selectbox(f'Выберите правильный ответ:', [*option] , key=f'phrase_{i}')
                check_button = st.button(f'Проверить', key=f'bphrase{i}')
    
                if check_button:
                    if user_answer.lower() == answ.lower():
                        st.success('Правильный ответ!')
                    else:
                        st.error('Неправильный ответ!')
            #elif task == 'select_sent':
               
                #rad = st.radio('Выберите правильное предложение:',['', *option], key=f'radio_{i}')
                
                #if rad == '':
                    #pass
                #elif rad == answ:
                    #st.success('Правильный ответ!')
               #else:
                    #st.error('Неправильный ответ!')
    

            


if __name__ == '__main__':
    nlp = en_core_web_sm.load()
    st.title("Генератор упражнений по английскому")
    text = st.text_area("Введите текст:", key="text_area")
    num = st.number_input("Введите количество упражнений:", min_value=1, step=1, key='num')
    main(text, num, nlp)
