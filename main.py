import polars as pl
from src.data.dataset import DataSet
from src.data.preprocess import PreProcess
from src.config import Config
# from src.model.model import Model


def init_dataset() -> pl.DataFrame:
    dataset: DataSet = DataSet(
        path='src/data/files/students_sao_paulo.csv',
    )
    df: pl.DataFrame = dataset.get()

    return df.preprocess.apply()

def init_model() -> Model:
    # model: Model = Model()

    # X_train, X_test, y_train, y_test = processed_dataset
    # model.set(input_shape=X_train.shape)
    # model.compile()
    # hist = model.train(X_train=X_train, y_train=y_train)



def main():

    # model: Model = Model()

    # X_train, X_test, y_train, y_test = processed_dataset
    # model.set(input_shape=X_train.shape)
    # model.compile()
    # hist = model.train(X_train=X_train, y_train=y_train)
    #
    # print(model.evaluate(X_test=X_test, y_test=y_test))


if __name__ == '__main__':
    main()
