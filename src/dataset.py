import json
import polars as pl
from utils.new_column import NewColumn


class DataSet:
    def __init__(self, path: str, config_path: str) -> None:
        self.__df: pl.DataFrame
        self.__config: dict
        self.__path: str = path
        self.__config_path: str = config_path

    def get(self) -> pl.DataFrame | None:
        return self.__df

    def load(self) -> None:
        self.__df = pl.read_csv(self.__path, separator=',')

        with open(self.__config_path, 'r') as f:
            self.__config = json.load(f)

    def preprocess(self) -> None:
        if len(self.__config.get('drop_columns', [])) > 0:
            self.__df = self.__df.drop(self.__config.get('drop_columns'))

        if 'birthdate' in self.__df.columns:
            self.__df = self.__df.new_column.age()

        for col in self.__config.get('categorical_columns', []):
            self.__df = self.__df.with_columns(
                pl.col(col).cast(pl.Categorical).alias(f'{col}_encoded')
            )
        


if __name__ == '__main__':
    dataset = DataSet(
        path='./files/students_sao_paulo.csv',
        config_path='./files/config.json'
    )

    dataset.load()
    dataset.preprocess()
    print(dataset.get())
