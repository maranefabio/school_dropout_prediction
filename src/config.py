from enum import Enum


class Config(Enum):
    TARGET: str = "dropout"
    DROP: list[str] = ("id", "name")
    FILL_METHOD: str = "mean"
    CATHEGORICAL: list[str] = (
        "gender",
        "race",
        "region"
    )
    NUMERICAL: list[str] = (
        "age",
        "net_income",
        "average_grades",
        "num_bathrooms",
        "distance_from_school"
    )
    TEST_PERCENT: int = 20
    SEED: int = 0


if __name__ == '__main__':
    pass
