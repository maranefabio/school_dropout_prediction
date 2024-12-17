import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.polars_extension.new_column import NewColumn
from src.config import Config


@pl.api.register_dataframe_namespace('preprocess')
class PreProcess:
    def __init__(self, df: pl.DataFrame) -> None:
        self.__df: pl.DataFrame = df

    def apply(self) -> pl.DataFrame:
        df: pl.DataFrame = self.__df

        df = self.clean_and_transform(df)
        df = self.encode(df)
        df = self.scale(df, how='min_max_normalization')

        return df

    def clean_and_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if len(Config.DROP.value) > 0:
            df = df.drop(Config.DROP.value)

        if 'birthdate' in df.columns:
            df = df.new_column.age()

        return df

    @pl.StringCache()
    def encode(self, df: pl.DataFrame) -> pl.DataFrame:
        for col in Config.CATHEGORICAL.value:
            df = df.with_columns(
                pl.col(col).cast(pl.Categorical).to_physical()
            )

        return df

    def scale(self,
              df: pl.DataFrame,
              how: str = 'min_max_normalization') -> pl.DataFrame:
        def min_max_normalization(df: pl.DataFrame) -> pl.DataFrame:
            for col in Config.NUMERICAL.value:
                df = df.with_columns(
                    (pl.col(col) - pl.col(col).min()) /
                    (pl.col(col).max() - pl.col(col).min())
                )
            return df

        def standardization(df: pl.DataFrame) -> pl.DataFrame:
            for col in Config.NUMERICAL.value:
                df = df.with_columns(
                    (pl.col(col) - pl.col(col).mean()) /
                    (pl.col(col).std())
                )
            return df

        match how:
            case 'min_max_normalization':
                return min_max_normalization(df)
            case 'standardization':
                return standardization(df)
            case _:
                return
