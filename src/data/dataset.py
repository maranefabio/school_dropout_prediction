import polars as pl
import json


class DataSet:
    def __init__(self, path: str, config_path: str) -> None:
        self.__path: pl.DataFrame = pl.read_csv(path, separator=','),
        self.__config_path: dict = json.load(config_path, 'r')

    def get_df(self) -> pl.DataFrame | None:
        return self.__df

    def get_config(self) -> dict | None:
        return self.__config


if __name__ == '__main__':
    pass
