import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import numpy as np

# GPU configuration snippet
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


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
        for i in range(num_models):
            self.models.append(self.build_model())

        # Create an instance of the AGI class
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
            model.fit(padded_sequences, y_train, epochs=epochs, batch_size=batch_size)

# Load the IMDb dataset
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Load and preprocess the CIFAR-10 dataset
(train_images, train_image_labels), (test_images, test_image_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Use the Keras built-in tokenizer and create the reverse word index
word_index = tf.keras.datasets.imdb.get_word_index()
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

# Train the WorkProcessAGI model
num_models = 2  # The number of models you want to create
work_process_agi = WorkProcessAGI(tokenizer, num_models)

# Calculate the AGI input_data for each review
agi_input_data = [np.mean(sequence) for sequence in train_data]

# Train the AGI text model
agi_text_X_train = np.array(agi_input_data).reshape(-1, 1)
agi_text_y_train = np.array(train_labels)
work_process_agi.agi.text_model.fit(agi_text_X_train, agi_text_y_train, epochs=10, batch_size=16)

# Train the AGI image model
agi_image_X_train = train_images
agi_image_y_train = train_image_labels
work_process_agi.agi.image_model.fit(agi_image_X_train, agi_image_y_train, epochs=10, batch_size=16)

# Train the WorkProcessAGI models
work_process_agi.fit(train_data_decoded, train_labels)



