import config
import openai
import requests
import json
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image
from io import BytesIO

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

openai.api_key = config.OPENAI_API_KEY

class AGI:
    def __init__(self, text_input_shape, image_input_shape, output_shape):
        self.text_input_shape = text_input_shape
        self.image_input_shape = image_input_shape
        self.output_shape = output_shape
        self.text_model = self.build_text_model()
        self.image_model = self.build_image_model()

    def build_text_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=self.text_input_shape),
            tf.keras.layers.Dense(self.output_shape, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def build_image_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.image_input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(self.output_shape, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

class WorkProcessAGI:
    def __init__(self, tokenizer, num_models):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.num_words
        self.max_length = 500
        self.models = []
        self.strategy = tf.distribute.MirroredStrategy()
        with self.strategy.scope():
            for i in range(num_models):
                self.models.append(self.build_model())

        self.agi = AGI((1,), (32, 32, 3), 1)

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32),
            tf.keras.layers.Embedding(self.vocab_size, 64),
            tf.keras.layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train, epochs=10, batch_size=16):
        padded_sequences = self.tokenizer.texts_to_sequences(X_train)
        padded_sequences = pad_sequences(padded_sequences, maxlen=self.max_length, padding='post',
                                         truncating='post')
        for model in self.models:
            with self.strategy.scope():
                model.fit(padded_sequences, y_train, epochs=epochs, batch_size=batch_size)

    def fit_both_models(self, text_X_train, text_y_train, image_X_train, image_y_train, epochs=10, batch_size=16):
        with self.strategy.scope():
            self.agi.text_model.fit(text_X_train, text_y_train, epochs=epochs, batch_size=batch_size)
            self.agi.image_model.fit(image_X_train, image_y_train, epochs=epochs, batch_size=batch_size)







