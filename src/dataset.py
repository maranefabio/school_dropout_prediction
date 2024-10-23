import json
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.new_column import NewColumn


class DataSet:
    def __init__(self, path: str, config_path: str) -> None:
        self.__df: pl.DataFrame
        self.__data: list[pl.DataFrame | pl.Series]
        self.__config: dict
        self.__path: str = path
        self.__config_path: str = config_path

    def get_df(self) -> pl.DataFrame | None:
        return self.__df

    def get_data(self) -> list[pl.DataFrame | pl.Series] | None:
        return self.__data

    def load(self) -> None:
        self.__df = pl.read_csv(self.__path, separator=',')

        with open(self.__config_path, 'r') as f:
            self.__config = json.load(f)

    @pl.StringCache()
    def preprocess(self) -> list[pl.DataFrame | pl.Series]:
        if len(self.__config.get('drop_columns', [])) > 0:
            self.__df = self.__df.drop(self.__config.get('drop_columns'))

        if 'birthdate' in self.__df.columns:
            self.__df = self.__df.new_column.age()

        for col in self.__config.get('categorical_columns', []):
            self.__df = self.__df.with_columns(
                pl.col(col).cast(pl.Categorical).to_physical()
            )

        self.__X = self.__df.drop(self.__config.get('target_column'))
        self.__y = self.__df[self.__config.get('target_column')]

        splitted_data: list = train_test_split(
            self.__X.to_numpy(),
            self.__y.to_numpy(),
            test_size=0.2,
            random_state=0
        )

        scaler = StandardScaler()
        for i in range(2):
            splitted_data[i] = scaler.fit_transform(splitted_data[i])

        self.__data = splitted_data


if __name__ == '__main__':
    dataset = DataSet(
        path='./files/students_sao_paulo.csv',
        config_path='./files/config.json'
    )

    dataset.load()
    dataset.preprocess(test_size=0.2, random_state=0)
    print(dataset.get_data())
