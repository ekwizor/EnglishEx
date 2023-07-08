import streamlit as st
import re
import random

def generate_exercise(sentence):
    words = sentence.split()
    num_words = len(words)
    
    # Генерируем индекс пропущенного слова (случайное число между 1 и предпоследним словом)
    missing_word_index = random.randint(1, num_words - 2)
    
    # Сохраняем пропущенное слово
    missing_word = words[missing_word_index]
    
    # Заменяем пропущенное слово символами подчеркивания
    words[missing_word_index] = '______'
    
    # Формируем упражнение
    exercise = ' '.join(words)
    
    return exercise, missing_word

# Настройка внешнего вида Streamlit
st.set_page_config(page_title='English Exercise Generator')

# Заголовок приложения
st.title('English Exercise Generator')

# Получение текста от пользователя
text = st.text_area('Введите текст для генерации упражнений', height=300)

# Генерация упражнений при нажатии кнопки
if st.button('Сгенерировать упражнение'):
    # Разбиваем текст на предложения
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    # Генерируем случайное предложение из введенного текста
    random_sentence = random.choice(sentences)
    
    # Генерируем упражнение
    exercise, missing_word = generate_exercise(random_sentence)
    
    # Отображение сгенерированного упражнения
    st.header('Упражнение:')
    st.write(exercise)
    
    # Получение ответа от пользователя
    user_answer = st.text_input('Введите пропущенное слово')
    
    # Проверка ответа пользователя
    if user_answer.lower() == missing_word.lower():
        st.write('Верно! Ответ:', missing_word)
    else:
        st.write('Неверно! Правильный ответ:', missing_word)

    
