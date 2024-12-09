import polars as pl
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


class Model:
    def __init__(self) -> None:
        pass

    def set(self, input_shape: tuple) -> None:
        model: tf.keras.models = Sequential([
            Dense(units=64, activation='relu'),
            Dense(units=32, activation='relu'),
            Dense(units=1, activation='sigmoid'),
        ])

        self.__model = model

    def compile(self) -> None:
        self.__model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train: np.array, y_train: np.array) -> None:
        return self.__model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

    def evaluate(self, X_test: np.array, y_test: np.array) -> list:
        return self.__model.evaluate(
            X_test,
            y_test
        )
