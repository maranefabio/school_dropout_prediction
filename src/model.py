import polars as pl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizer import Adam


class Model:
    def __init__(
        self,
        X_train: pl.DataFrame,
        X_test: pl.DataFrame,
        y_train: pl.Series,
        y_test: pl.Series
    ) -> None:
        self.__X_train = X_train
        self.__X_test = X_test
        self.__y_train = y_train
        self.__y_test = y_test
        self.__model: tf.keras.models

    def set(self) -> None:
        model: tf.keras.models = Sequential()

        model.add(Dense(
            units=64,
            activation='relu',
            input_shape=self.get_X.shape[1]
        ))
        model.add(Dense(
            units=32,
            activation='relu'
        ))
        model.add(Dense(
            units=1,
            activation='sigmoid'
        ))

        self.__model = model

    def compile(self) -> None:
        self.__model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self) -> None:
        self.__model.fit(
            self.__X_train,
            self.__y_test,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

    def evaluate(self):
        return self.__model.evaluate(
            self.__X_test,
            self.__y_test
        )
