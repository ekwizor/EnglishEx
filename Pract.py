import streamlit as st
import random

def generate_exercises(text, num_exercises):
    # Разделяем текст на предложения
    sentences = text.split('. ')
    selected_sentences = random.sample(sentences, num_exercises)
    
    exercises = []
    for sentence in selected_sentences:
        # Удаляем знаки препинания
        sentence = sentence.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
        # Разделяем предложение на слова
        words = sentence.split(' ')
        # Выбираем случайное слово в предложении
        selected_word = random.choice(words)
        
        exercise = f"Замените слово '{selected_word}' в предложении: {sentence}"
        exercises.append(exercise)
    
    return exercises

def main():
    st.title("Генератор упражнений по английскому языку")
    
    # Ввод текста пользователем
    text = st.text_area("Введите текст", "Example sentence 1. Example sentence 2. Example sentence 3.")
    
    # Ввод количества упражнений пользователем
    num_exercises = st.slider("Выберите количество упражнений", 1, 10)
    
    # Генерация упражнений по нажатию кнопки
    if st.button("Сгенерировать упражнения"):
        exercises = generate_exercises(text, num_exercises)
        for exercise in exercises:
            st.write(exercise)

if __name__ == "__main__":
    main()
