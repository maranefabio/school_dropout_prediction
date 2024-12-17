import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.polars_extension.new_column import NewColumn
from src.config import Config


class PreProcess:
    def __init__(self, df: pl.DataFrame) -> None:
        self.__df: pl.DataFrame = df
        self.__config: dict = df.get_config()

    def clean_and_transform(self) -> None:
        df: pl.DataFrame = self.__df
        if len(self.__config.get('DROP')) > 0:
            df = df.drop(self.__config.get('DROP'))

        if 'birthdate' in df.columns:
            df = df.new_column.age()

        self.__df = df

    @pl.StringCache()
    def encode(self) -> None:
        df: pl.DataFrame = self.__df
        for col in self.__config.get('CATHEGORICAL'):
            df = df.with_columns(
                pl.col(col).cast(pl.Categorical).to_physical()
            )

        self.__df = df

    def scale(self, how: str = 'min_max_normalization') -> None:
        df: pl.DataFrame = self.__df

        def min_max_normalization(df: pl.DataFrame) -> pl.DataFrame:
            for col in self.__config.get('NUMERICAL'):
                df = df.with_columns(
                    (pl.col(col) - pl.col(col).min()) /
                    (pl.col(col).max() - pl.col(col).min())
                )
            return df

        def standardization(df: pl.DataFrame) -> pl.DataFrame:
            for col in self.__config.get('NUMERICAL'):
                df = df.with_columns(
                    (pl.col(col) - pl.col(col).mean()) /
                    (pl.col(col).std())
                )
            return df

        match how:
            case 'min_max_normalization':
                self.__df = min_max_normalization(df)
            case 'standardization':
                self.__df = standardization(df)
            case _:
                return
