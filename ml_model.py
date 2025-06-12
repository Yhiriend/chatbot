import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os

class MLModel:
    def __init__(self, input_shape, output_shape, model_path):
        self.model = Sequential([
            Dense(100, activation='relu', input_shape=(input_shape,)),
            Dense(50),
            Dropout(0.5),
            Dense(output_shape, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model_path = model_path

    def load_or_train(self, training_data, output_data):
        if os.path.isfile(self.model_path):
            self.model.load_weights(self.model_path)
        else:
            self.model.fit(training_data, output_data, validation_split=0.1, epochs=1000, batch_size=128)
            self.model.save_weights(self.model_path)

    def predict(self, data):
        return self.model.predict(data) 