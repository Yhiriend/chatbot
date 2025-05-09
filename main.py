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
import requests
import os
import matplotlib as mtpl
import dload

ignore_words = ['?', '!', '.', ',', '¿', '¡', "'", '"', ':', ';', '(', ')', '[', ']', '{', '}', '|', '\\', '/', '`', '~', '*', '#', '^', '_', '=', '+', '-', '>', '<', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Solo clonar el repositorio si no existe
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

# Crear el modelo usando Keras
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

def response(sentence):
    if sentence == "salir":
        print("Chao")
        return False
    else:
        bucket = [0 for _ in range(len(all_words))]
        processed_sentence = nltk.word_tokenize(sentence)
        processed_sentence = [stemmer.stem(word.lower()) for word in processed_sentence if word not in ignore_words]
        for word in processed_sentence:
            for i, w in enumerate(all_words):
                if w == word:
                    bucket[i] = 1
        results = model.predict(np.array([bucket]))
        results_index = np.argmax(results)
        max = results[0][results_index]

        target = tags[results_index]

        for tg in database["intents"]:
            if tg["tag"] == target:
                responses = tg["responses"]
                response = random.choice(responses)
        if max > 0.7:
            print("Bot:", response)
            return True
        else:
            print("Bot: No entiendo")
            return True

print("Bienvenido al bot de chat de la empresa")
while True:
    sentence = input("Tú: ")
    if not response(sentence):
        break



