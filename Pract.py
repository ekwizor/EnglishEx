import streamlit as st
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import random

def generate_exercises(text):
    # Разделяем текст на предложения
    sentences = sent_tokenize(text)

    exercises = []
    for sentence in sentences:
        # Токенизация слов
        words = word_tokenize(sentence.lower())

        # Удаление стоп-слов и пунктуации
        words = [word for word in words if word not in stopwords.words("english") and word not in punctuation]

        # Проверка, что предложение содержит достаточно слов
        if len(words) >= 3:
            # Выбираем случайное слово из предложения
            target_word = random.choice(words)

            # Заменяем выбранное слово символами подчеркивания
            exercise_sentence = sentence.replace(target_word, "______")

            exercises.append((exercise_sentence, target_word))

    return exercises


# Основной код приложения
def main():
    # Заголовок и описание приложения
    st.title("Генератор упражнений по английскому языку")
    st.write("Введите текст, чтобы сгенерировать упражнения.")

    # Поле для ввода текста
    user_text = st.text_area("Введите текст")

    # Кнопка для генерации упражнений
    if st.button("Сгенерировать упражнения"):
        exercises = generate_exercises(user_text)

        if len(exercises) > 0:
            st.subheader("Упражнения:")
            for exercise in exercises:
                st.write("- " + exercise[0])
                st.write("Ответ: " + exercise[1])

        else:
            st.write("Нет подходящих предложений для генерации упражнений.")

if __name__ == "__main__":
    main()
