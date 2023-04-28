import tensorflow as tf
import numpy as np
import string

# GPU configuration snippet
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class WorkProcessAGI:
    def __init__(self, tokenizer, num_models):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.max_length = tokenizer.max_length
        self.models = []
        for i in range(num_models):
            self.models.append(self.build_model())

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
        padded_sequences = self.tokenizer.pad_sequences(X_train, maxlen=self.max_length, padding='post', truncating='post')
        for model in self.models:
            model.fit(padded_sequences, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, sequence, model_idx):
        input_sequence = self.tokenizer.texts_to_sequences([sequence])
        padded_sequence = self.tokenizer.pad_sequences(input_sequence, maxlen=self.max_length, padding='post', truncating='post')
        predicted_label = self.models[model_idx].predict(padded_sequence)[0][0]
        return predicted_label

    def learn_from_game(self, game_data, gpu=False):
        # Train the AGI model using game data
        if gpu:
            with tf.device('/GPU:1'):
                self.fit(game_data['X_train'], game_data['y_train'])
        else:
            self.fit(game_data['X_train'], game_data['y_train'])

    def manage_work_process(self, task_list, num_autonomous_models=0, game_data=None, gpu=False):
        # Define the work process management task
        # Here's an example of shuffling the task list and assigning them to different models for processing
        np.random.shuffle(task_list)
        for i, task in enumerate(task_list):
            if i < num_autonomous_models:
                # Autonomous models can handle tasks on their own
                self.predict(task, i)
            else:
                # Assist other models with their tasks
                model_idx = (i - num_autonomous_models) % len(self.models)
                self.predict(task, model_idx)

        if game_data is not None:
            # Train AGI model with game data
            self.learn_from_game(game_data, gpu)
