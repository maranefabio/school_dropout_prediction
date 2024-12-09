import polars as pl
import datetime as dt


@pl.api.register_dataframe_namespace('new_column')
class NewColumn:
    def __init__(self, df: pl.DataFrame) -> None:
        self.__df = df

    def age(self) -> list[pl.DataFrame]:
        return self.__df.with_columns(
            age=((
                dt.date.today() - pl.col('birthdate').cast(pl.Date)
            ).dt.total_days() / 365.25).cast(pl.Int32)
        ).drop('birthdate')
