import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import json
import random
import pickle
import os
import dload
import flet as ft
import time
import threading

ignore_words = ['?', '!', '.', ',', '¿', '¡', "'", '"', ':', ';', '(', ')', '[', ']', '{', '}', '|', '\\', '/', '`', '~', '*', '#', '^', '_', '=', '+', '-', '>', '<', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

if not os.path.exists("data_bot"):
    dload.git_clone("https://github.com/boomcrash/data_bot.git")

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace("\\", "//")
with open(dir_path + "/data_bot/data_bot-main/data.json", "r") as file:
    database = json.load(file)

words = []
all_words = []
tags = []
aux = []
auxA = []
auxB = []
training = []
output = []

try:
    with open("Entranamiento/brain.pickle", "rb") as pickleBrain:
        all_words, tags, training, output = pickle.load(pickleBrain)
except:
    for intent in database["intents"]:
        for pattern in intent["patterns"]:
            auxWords = nltk.word_tokenize(pattern)
            auxA.append(auxWords)
            auxB.append(auxWords)
            aux.append(intent["tag"])
    for w in auxB:
        if w not in ignore_words:
            words.append(w)
    import itertools
    words = sorted(set(list(itertools.chain.from_iterable(words))))
    tags = sorted(set(aux))
    all_words=[stemmer.stem(w.lower()) for w in words]
    all_words = sorted(list(set(all_words)))
    tags=sorted(tags)

    null_output = [0 for _ in range(len(tags))]

    for i, doc in enumerate(auxA):
        bag = []
        auxWords = [stemmer.stem(w.lower()) for w in doc if w !="?"]
        for w in all_words:
            if w in auxWords:
                bag.append(1)
            else:
                bag.append(0)
        output_row = null_output[:]        
        output_row[tags.index(aux[i])] = 1
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("Entranamiento/brain.pickle", "wb") as pickleBrain:
        pickle.dump((all_words, tags, training, output), pickleBrain)

model = Sequential([
    Dense(100, activation='relu', input_shape=(len(training[0]),)),
    Dense(50),
    Dropout(0.5),
    Dense(len(output[0]), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Cargar o entrenar el modelo
if os.path.isfile(dir_path + "/Entranamiento/model.weights.h5"):
    model.load_weights(dir_path + "/Entranamiento/model.weights.h5")
else:
    model.fit(training, output, validation_split=0.1, epochs=1000, batch_size=128)
    model.save_weights(dir_path + "/Entranamiento/model.weights.h5")

def get_bot_response(sentence):
    bucket = [0 for _ in range(len(all_words))]
    processed_sentence = nltk.word_tokenize(sentence)
    processed_sentence = [stemmer.stem(word.lower()) for word in processed_sentence if word not in ignore_words]
    for word in processed_sentence:
        for i, w in enumerate(all_words):
            if w == word:
                bucket[i] = 1
    results = model.predict(np.array([bucket]))
    results_index = np.argmax(results)
    max_prob = results[0][results_index]
    target = tags[results_index]
    for tg in database["intents"]:
        if tg["tag"] == target:
            responses = tg["responses"]
            response = random.choice(responses)
    if max_prob > 0.7:
        return response
    else:
        return "Lo siento, no entiendo tu pregunta. ¿Podrías reformularla?"

def main(page: ft.Page):
    page.title = "Chatbot Asistente"
    page.bgcolor = "#F7F9FA"
    page.window_width = 500
    page.window_height = 700
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    chat_column = ft.Column(scroll=ft.ScrollMode.ALWAYS, expand=True, spacing=10)

    input_field = ft.TextField(
        hint_text="Escribe tu mensaje...",
        expand=True,
        border_radius=20,
        bgcolor="white",
        border_color="#4A90E2",
        color="black",
        autofocus=True
    )

    def send_message(e=None):
        user_msg = input_field.value.strip()
        if user_msg:
            # Mensaje del usuario (alineado a la derecha)
            chat_column.controls.append(
                ft.Row([
                    ft.Container(
                        content=ft.Text(user_msg, color="black", selectable=True),
                        bgcolor="#E3E8F0",
                        border_radius=20,
                        padding=10,
                        margin=0,
                        alignment=ft.alignment.center_right,
                        width=300,
                    )
                ], alignment=ft.MainAxisAlignment.END)
            )
            page.update()
            input_field.value = ""
            page.update()

            # Animación de puntos suspensivos (escribiendo...)
            typing_container = ft.Container(
                content=ft.Text(".", color="black"),
                bgcolor="white",
                border_radius=20,
                padding=10,
                margin=0,
                alignment=ft.alignment.center_left,
                width=60,
            )
            typing_row = ft.Row([typing_container], alignment=ft.MainAxisAlignment.START)
            chat_column.controls.append(typing_row)
            page.update()

            def bot_typing_animation():
                dots = [".", "..", "..."]
                for i in range(6):  # 2 segundos aprox
                    typing_container.content.value = dots[i % 3]
                    page.update()
                    time.sleep(0.33)
                # Obtener respuesta real del bot
                bot_msg = get_bot_response(user_msg)
                # Reemplazar animación por respuesta real
                chat_column.controls.remove(typing_row)
                chat_column.controls.append(
                    ft.Row([
                        ft.CircleAvatar(
                            content=ft.Text("🤖", size=20),
                            bgcolor="#4A90E2",
                            radius=20,
                        ),
                        ft.Container(
                            content=ft.Text(bot_msg, color="black", selectable=True),
                            bgcolor="white",
                            border_radius=20,
                            padding=10,
                            margin=0,
                            alignment=ft.alignment.center_left,
                            width=300,
                        )
                    ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START)
                )
                page.update()

            threading.Thread(target=bot_typing_animation, daemon=True).start()

    send_btn = ft.IconButton(
        icon="send",
        bgcolor="#4A90E2",
        icon_color="white",
        on_click=send_message,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=20))
    )

    page.add(
        ft.Container(
            content=ft.Column([
                ft.Text("Chatbot Asistente", size=24, weight=ft.FontWeight.BOLD, color="#4A90E2"),
                ft.Container(chat_column, expand=True, height=500, bgcolor="#F7F9FA", border_radius=20, padding=10),
                ft.Row([input_field, send_btn], alignment=ft.MainAxisAlignment.END)
            ], expand=True),
            padding=20,
            expand=True
        )
    )
    input_field.on_submit = send_message

if __name__ == "__main__":
    ft.app(target=main)



