import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.polars_extension.new_column import NewColumn
from src.config import Config


class PreProcess:
    def __init__(self, raw_df: pl.DataFrame, config: Config) -> None:
        self.__df: pl.DataFrame = raw_df
        self.__config: Config = config
        self.__data: list[pl.DataFrame | pl.Series] = None

    def get_df(self) -> pl.DataFrame:
        return self.__df

    def get_data(self) -> pl.DataFrame:
        return self.__data

    @pl.StringCache()
    def clean_and_transform(self) -> None:
        df: pl.DataFrame = self.__df

        if len(self.__config.DROP.value) > 0:
            df = df.drop(self.__config.DROP.value)

        if 'birthdate' in self.__df.columns:
            df = df.new_column.age()

        self.__df = df

    def encode(self) -> None:
        df: pl.DataFrame = self.__df

        for col in self.__config.CATHEGORICAL.value:
            df = df.with_columns(
                pl.col(col).cast(pl.Categorical).to_physical()
            )
        self.__df = df

    def split(self) -> None:
        x = self.__df.drop(self.__config.TARGET.value)
        y = self.__df[self.__config.TARGET.value]

        data: list = train_test_split(
            x.to_numpy(),
            y.to_numpy(),
            test_size=self.__config.TEST_PERCENT.value / 100,
            random_state=self.__config.SEED.value
        )

        self.__data = data

    def scale(self) -> None:
        scaler: StandardScaler = StandardScaler()
        data: list[pl.Series | pl.DataFrame] = self.__data
        for i in range(2):
            data[i] = scaler.fit_transform(data[i])

        self.__data = data
