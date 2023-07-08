import streamlit as st
import random

exercises = [
    {
        "sentence": "The * is shining brightly in the sky.",
        "answer": "sun"
    },
    {
        "sentence": "She * to the store to buy some milk.",
        "answer": "went"
    },
    {
        "sentence": "I like to * books in my free time.",
        "answer": "read"
    },
    {
        "sentence": "My cat loves to * with a ball of yarn.",
        "answer": "play"
    },
    {
        "sentence": "He * his homework before going to bed.",
        "answer": "completes"
    }
]

def generate_exercise():
    exercise = random.choice(exercises)
    return exercise["sentence"]

def check_answer(exercise, user_answer):
    exercise_answer = next((item for item in exercises if item["sentence"] == exercise), None)
    return user_answer.lower() == exercise_answer["answer"].lower() if exercise_answer else False

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
    st.write(exercise.replace("*", "______"))

if __name__ == "__main__":
    main()
