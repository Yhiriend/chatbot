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
import codecs
import re

ignore_words = ['?', '!', '.', ',', '¿', '¡', "'", '"', ':', ';', '(', ')', '[', ']', '{', '}', '|', '\\', '/', '`', '~', '*', '#', '^', '_', '=', '+', '-', '>', '<', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

if not os.path.exists("data_bot"):
    dload.git_clone("https://github.com/boomcrash/data_bot.git")

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace("\\", "//")

# Cargar ambos archivos JSON con codificación UTF-8
with codecs.open(dir_path + "/data_bot/data_bot-main/conceptos.json", "r", encoding='utf-8') as file:
    database_conceptos = json.load(file)
with codecs.open(dir_path + "/data_bot/data_bot-main/ejemplos.json", "r", encoding='utf-8') as file:
    database_ejemplos = json.load(file)

words = []
all_words = []
tags_conceptos = []
tags_ejemplos = []
aux = []
auxA = []
auxB = []
training = []
output_conceptos = []
output_ejemplos = []

try:
    with open("Entranamiento/brain.pickle", "rb") as pickleBrain:
        all_words, tags_conceptos, tags_ejemplos, training, output_conceptos, output_ejemplos = pickle.load(pickleBrain)
except:
    # Procesar conceptos
    for intent in database_conceptos["intents"]:
        for pattern in intent["patterns"]:
            auxWords = nltk.word_tokenize(pattern)
            auxA.append(auxWords)
            auxB.append(auxWords)
            aux.append(intent["tag"])
            tags_conceptos.append(intent["tag"])

    # Procesar ejemplos
    for intent in database_ejemplos["intents"]:
        for pattern in intent["patterns"]:
            auxWords = nltk.word_tokenize(pattern)
            auxA.append(auxWords)
            auxB.append(auxWords)
            aux.append(intent["tag"])
            tags_ejemplos.append(intent["tag"])

    for w in auxB:
        if w not in ignore_words:
            words.append(w)

    import itertools
    words = sorted(set(list(itertools.chain.from_iterable(words))))
    tags_conceptos = sorted(set(tags_conceptos))
    tags_ejemplos = sorted(set(tags_ejemplos))
    all_words = [stemmer.stem(w.lower()) for w in words]
    all_words = sorted(list(set(all_words)))

    null_output_conceptos = [0 for _ in range(len(tags_conceptos))]
    null_output_ejemplos = [0 for _ in range(len(tags_ejemplos))]

    for i, doc in enumerate(auxA):
        bag = []
        auxWords = [stemmer.stem(w.lower()) for w in doc if w != "?"]
        for w in all_words:
            if w in auxWords:
                bag.append(1)
            else:
                bag.append(0)
        
        # Crear vectores de salida para conceptos y ejemplos
        output_row_conceptos = null_output_conceptos[:]
        output_row_ejemplos = null_output_ejemplos[:]
        
        if aux[i] in tags_conceptos:
            output_row_conceptos[tags_conceptos.index(aux[i])] = 1
        if aux[i] in tags_ejemplos:
            output_row_ejemplos[tags_ejemplos.index(aux[i])] = 1
            
        training.append(bag)
        output_conceptos.append(output_row_conceptos)
        output_ejemplos.append(output_row_ejemplos)

    training = np.array(training)
    output_conceptos = np.array(output_conceptos)
    output_ejemplos = np.array(output_ejemplos)

    with open("Entranamiento/brain.pickle", "wb") as pickleBrain:
        pickle.dump((all_words, tags_conceptos, tags_ejemplos, training, output_conceptos, output_ejemplos), pickleBrain)

# Crear dos modelos: uno para conceptos y otro para ejemplos
model_conceptos = Sequential([
    Dense(100, activation='relu', input_shape=(len(training[0]),)),
    Dense(50),
    Dropout(0.5),
    Dense(len(output_conceptos[0]), activation='softmax')
])

model_ejemplos = Sequential([
    Dense(100, activation='relu', input_shape=(len(training[0]),)),
    Dense(50),
    Dropout(0.5),
    Dense(len(output_ejemplos[0]), activation='softmax')
])

model_conceptos.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

model_ejemplos.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

# Cargar o entrenar los modelos
if os.path.isfile(dir_path + "/Entranamiento/model_conceptos.weights.h5"):
    model_conceptos.load_weights(dir_path + "/Entranamiento/model_conceptos.weights.h5")
else:
    model_conceptos.fit(training, output_conceptos, validation_split=0.1, epochs=1000, batch_size=128)
    model_conceptos.save_weights(dir_path + "/Entranamiento/model_conceptos.weights.h5")

if os.path.isfile(dir_path + "/Entranamiento/model_ejemplos.weights.h5"):
    model_ejemplos.load_weights(dir_path + "/Entranamiento/model_ejemplos.weights.h5")
else:
    model_ejemplos.fit(training, output_ejemplos, validation_split=0.1, epochs=1000, batch_size=128)
    model_ejemplos.save_weights(dir_path + "/Entranamiento/model_ejemplos.weights.h5")

def get_bot_response(sentence):
    bucket = [0 for _ in range(len(all_words))]
    processed_sentence = nltk.word_tokenize(sentence)
    processed_sentence = [stemmer.stem(word.lower()) for word in processed_sentence if word not in ignore_words]
    
    for word in processed_sentence:
        for i, w in enumerate(all_words):
            if w == word:
                bucket[i] = 1

    # Obtener predicciones de ambos modelos
    results_conceptos = model_conceptos.predict(np.array([bucket]))
    results_ejemplos = model_ejemplos.predict(np.array([bucket]))
    
    results_index_conceptos = np.argmax(results_conceptos)
    results_index_ejemplos = np.argmax(results_ejemplos)
    
    max_prob_conceptos = results_conceptos[0][results_index_conceptos]
    max_prob_ejemplos = results_ejemplos[0][results_index_ejemplos]
    
    respuesta = ""
    
    # Obtener respuesta de conceptos si la probabilidad es alta
    if max_prob_conceptos > 0.7:
        target_concepto = tags_conceptos[results_index_conceptos]
        for tg in database_conceptos["intents"]:
            if tg["tag"] == target_concepto:
                respuesta += "📚 **Concepto:**\n"
                respuesta += random.choice(tg["responses"]) + "\n\n"
    
    # Obtener respuesta de ejemplos si la probabilidad es alta
    if max_prob_ejemplos > 0.7:
        target_ejemplo = tags_ejemplos[results_index_ejemplos]
        for tg in database_ejemplos["intents"]:
            if tg["tag"] == target_ejemplo:
                respuesta += "💻 **Ejemplo Práctico:**\n"
                respuesta += random.choice(tg["responses"])
    
    if not respuesta:
        return "Lo siento, no entiendo tu pregunta. ¿Podrías reformularla?"
    
    return respuesta

def main(page: ft.Page):
    page.title = "Chatbot Especialista en Funciones Anónimas"
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
                # Reemplazar animación por respuesta real, mostrando código con ft.Markdown
                chat_column.controls.remove(typing_row)

                # Separar texto normal y bloque de código
                code_block = None
                texto_normal = bot_msg
                match = re.search(r'```(.*?)```', bot_msg, re.DOTALL)
                if match:
                    code_block = match.group(0)
                    texto_normal = bot_msg.replace(code_block, '').strip()
                    code_block = code_block.strip('`')  # Quitar las comillas invertidas

                # Mostrar texto normal con efecto typing palabra por palabra
                bot_text = ft.Text("", color="black", selectable=True)
                bot_row = ft.Row([
                    ft.CircleAvatar(
                        content=ft.Text("🤖", size=20),
                        bgcolor="#4A90E2",
                        radius=20,
                    ),
                    ft.Container(
                        content=ft.Column([bot_text], tight=True),
                        bgcolor="white",
                        border_radius=20,
                        padding=10,
                        margin=0,
                        alignment=ft.alignment.center_left,
                        width=300,
                    )
                ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START)
                chat_column.controls.append(bot_row)
                page.update()
                palabras = texto_normal.split(" ")
                texto_actual = ""
                for palabra in palabras:
                    texto_actual += palabra + " "
                    bot_text.value = texto_actual
                    page.update()
                    time.sleep(0.12)
                bot_text.value = texto_actual.strip()
                page.update()

                # Si hay bloque de código, mostrarlo con ft.Markdown
                if code_block:
                    # Extraer el lenguaje y el código
                    code_match = re.match(r'(\w+)?\n([\s\S]*)', code_block)
                    if code_match:
                        lenguaje = code_match.group(1) or "typescript"
                        codigo = code_match.group(2)
                    else:
                        lenguaje = "typescript"
                        codigo = code_block
                    markdown_code = f"```{lenguaje}\n{codigo}\n```"
                    # Agregar el bloque de código debajo del texto
                    bot_row.controls[1].content.controls.append(
                        ft.Markdown(markdown_code, selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB, code_theme="atom-one-light")
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
                ft.Text("Chatbot Especialista en Funciones Anónimas", size=24, weight=ft.FontWeight.BOLD, color="#4A90E2"),
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



