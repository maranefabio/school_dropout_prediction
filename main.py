import polars as pl
from src.data.dataset import DataSet
from src.data.preprocess import PreProcess
from src.config import Config
# from src.model.model import Model


def init_dataset() -> pl.DataFrame:
    dataset: DataSet = DataSet(
        path='src/data/files/students_sao_paulo.csv',
        config='src/data/file/config.json'
    )

    return dataset.get_df()


def main() -> None:
    df: pl.DataFrame = init_dataset()


if __name__ == '__main__':
    main()
