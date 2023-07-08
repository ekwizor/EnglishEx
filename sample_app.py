import streamlit as st
import random

# Список упражнений
exercises = [
    "The * is shining brightly in the sky.",
    "She * to the store to buy some milk.",
    "I like to * books in my free time.",
    "My cat loves to * with a ball of yarn.",
    "He * his homework before going to bed."
]

def generate_exercise():
    exercise = random.choice(exercises)
    return exercise.replace("*", "______")

def check_answer(exercise, user_answer):
    answer = exercise.replace("______", "*")
    return user_answer.lower() == answer.lower()

# Основной код Streamlit
def main():
    st.title("English Exercise Generator")
    st.write("Введите правильное слово для каждого упражнения.")

    exercise = generate_exercise()
    user_answer = st.text_input("Заполните пропущенное слово:", value="", key="exercise_input")

    if st.button("Проверить"):
        if check_answer(exercise, user_answer):
            st.write("Правильно!")
        else:
            st.write("Неправильно! Попробуйте еще раз.")

    st.write("Упражнение:")
    st.write(exercise)


if __name__ == "__main__":
    main()
    
