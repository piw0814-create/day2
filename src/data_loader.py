"""Adult Census Income 데이터를 Pandas와 Polars로 불러오는 모듈."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import polars as pl
import yaml


COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def load_config(config_path: str = "config.yaml") -> dict:
    """YAML 설정 파일을 읽어 딕셔너리로 반환한다."""
    with open(config_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_data(url: str, raw_path: str) -> Path:
    """원본 데이터가 없을 경우 UCI 저장소에서 내려받는다."""
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        print("원본 데이터를 다운로드합니다.")
        urlretrieve(url, path)

    return path


def load_with_pandas(path: Path) -> pd.DataFrame:
    """Adult 데이터를 Pandas DataFrame으로 불러온다."""
    return pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )


def load_with_polars(path: Path) -> pl.DataFrame:
    """Adult 데이터를 Polars DataFrame으로 불러온다."""
    df = pl.read_csv(
        path,
        has_header=False,
        new_columns=COLUMNS,
        null_values=["?", " ?"],
    )
    # 파일 마지막의 빈 줄처럼 모든 컬럼이 null인 행만 제거한다.
    df = df.filter(pl.any_horizontal(pl.all().is_not_null()))  
    
    # 문자열 컬럼 앞뒤의 공백을 제거한다.
    string_columns = [
        column
        for column, dtype in zip(df.columns, df.dtypes)
        if dtype == pl.String
    ]

    return df.with_columns(pl.col(string_columns).str.strip_chars())


if __name__ == "__main__":
    config = load_config()
    data_path = download_data(
        config["data"]["url"],
        config["data"]["raw_path"],
    )

    pandas_df = load_with_pandas(data_path)
    polars_df = load_with_polars(data_path)

    print(f"Pandas shape: {pandas_df.shape}")
    print(f"Polars shape: {polars_df.shape}")
    print(f"컬럼 일치 여부: {pandas_df.columns.tolist() == polars_df.columns}")