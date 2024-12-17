import polars as pl


class DataSet:
    def __init__(self, path: str) -> None:
        self.__df = pl.read_csv(path, separator=',')

    def get(self) -> pl.DataFrame | None:
        return self.__df


if __name__ == '__main__':
    pass
