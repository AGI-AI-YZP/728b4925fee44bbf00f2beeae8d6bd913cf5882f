import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

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

    def fit(self, text_X_train, text_y_train, image_X_train, image_y_train, epochs=10, batch_size=16):
        self.text_model.fit(text_X_train, text_y_train, epochs=epochs, batch_size=batch_size)
        self.image_model.fit(image_X_train, image_y_train, epochs=epochs, batch_size=batch_size)

class AGIApp:
    def __init__(self, agi, tokenizer):
        self.agi = agi
        self.tokenizer = tokenizer
        self.root = tk.Tk()
        self.root.title("AGI Demo")

        self.chat_panel = scrolledtext.ScrolledText(self.root, wrap=tk.WORD)
        self.chat_panel.pack(expand=True, fill='both')

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(self.root, textvariable=self.entry_var, width=50)
        self.entry.bind('<Return>', self.send_message)
        self.entry.pack(side=tk.LEFT, fill='x', expand=True)

        self.send_button = tk.Button(self.root, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

        self.root.mainloop()

    def send_message(self, event=None):
        message = self.entry_var.get().strip()
        if message:
            self.entry_var.set('')
            self.chat_panel.configure(state='normal')
            self.chat_panel.insert(tk.END, f"You: {message}\n")
            self.chat_panel.configure(state='disabled')

            # Process the message with AGI
            response = self.process_message_with_agi(message)
            self.chat_panel.configure(state='normal')
            self.chat_panel.insert(tk.END, response)
            self.chat_panel.configure(state='disabled')
            self.chatself.chat_panel.see(tk.END)

    def preprocess_text(self, text):
        tokenized_text = self.tokenizer.texts_to_sequences([text])
        padded_text = pad_sequences(tokenized_text, maxlen=500, padding='post', truncating='post')
        return padded_text

    def process_message_with_agi(self, message):
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(message)

        if sentiment_score['compound'] >= 0.05:
            sentiment = "positive"
        elif sentiment_score['compound'] <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        response = f"AGI response: The sentiment of the input message is {sentiment}.\n"
        return response

if __name__ == "__main__":
    # Load and preprocess the IMDb dataset
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    # Load and preprocess the CIFAR-10 dataset
    (train_images, train_image_labels), (test_images, test_image_labels) = cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Use the Keras built-in tokenizer and create the reverse word index
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    # Decode the training data into text
    train_data_decoded = [" ".join([reverse_word_index.get(i, "<UNK>") for i in sequence]) for sequence in train_data]

    # Tokenize and pad the sequences
    tokenizer = Tokenizer(num_words=10000, oov_token="<UNK>")
    tokenizer.fit_on_texts(train_data_decoded)
    padded_sequences = tokenizer.texts_to_sequences(train_data_decoded)
    padded_sequences = pad_sequences(padded_sequences, maxlen=500, padding='post', truncating='post')

    # Calculate the AGI input_data for each review
    agi_input_data = np.array([np.mean(sequence) for sequence in padded_sequences])

    # Prepare the data for the AGI models
    agi_text_X_train = agi_input_data.reshape(-1, 1)
    agi_image_X_train = train_images
    text_y_train = np.array(train_labels)
    image_y_train = np.array(train_image_labels)  # Use separate target labels for image model

    # Create an instance of the AGI class and train the models
    agi = AGI((1,), (32, 32, 3), 1)
    agi.fit(agi_text_X_train, text_y_train, agi_image_X_train, image_y_train, epochs=10, batch_size=16)

    # Initialize the AGIApp with the trained AGI model and the tokenizer
    app = AGIApp(agi, tokenizer)
