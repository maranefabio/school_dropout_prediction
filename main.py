from src.dataset import DataSet
from src.model import Model

dataset = DataSet(
    path='src/files/students_sao_paulo.csv',
    config_path='src/files/config.json'
)

model = Model()


def main():
    dataset.load()
    dataset.preprocess()
    X_train, X_test, y_train, y_test = dataset.get_data()

    model.set(input_shape=X_train.shape)
    model.compile()
    hist = model.train(X_train=X_train, y_train=y_train)

    print(model.evaluate(X_test=X_test, y_test=y_test))


if __name__ == '__main__':
    main()
