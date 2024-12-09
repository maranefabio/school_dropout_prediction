import polars as pl


class DataSet:
    def __init__(self, path: str) -> None:
        self.__df: pl.DataFrame
        self.__config: dict
        self.__path: str = path

    def get(self) -> pl.DataFrame | None:
        return self.__df

    def load(self) -> None:
        self.__df = pl.read_csv(self.__path, separator=',')


if __name__ == '__main__':
    pass
