"""Adult Census Income 데이터를 Pandas와 Polars로 불러오고 비교하는 모듈."""

import timeit
from pathlib import Path
from statistics import mean
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

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
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
    df = df.filter(
        pl.any_horizontal(pl.all().is_not_null())
    )

    # 문자열 컬럼 앞뒤의 공백을 제거한다.
    string_columns = [
        column
        for column, dtype in zip(df.columns, df.dtypes)
        if dtype == pl.String
    ]

    df = df.with_columns(
        pl.col(string_columns).str.strip_chars()
    )

    # 공백 때문에 문자열로 인식된 수치형 컬럼을 정수형으로 변환한다.
    df = df.with_columns(
        pl.col(NUMERIC_COLUMNS).cast(
            pl.Int64,
            strict=False,
        )
    )

    return df

def normalize_dtype(dtype: object) -> str:
    """
    Pandas와 Polars의 자료형 이름을 공통 범주로 변환한다.

    예:
    Pandas int64와 Polars Int64는 모두 integer로 변환한다.
    Pandas object와 Polars String은 모두 string으로 변환한다.
    """
    dtype_name = str(dtype).lower()

    if dtype_name.startswith(("int", "uint")):
        return "integer"

    if dtype_name.startswith(("float", "decimal")):
        return "float"

    if dtype_name in {
        "object",
        "string",
        "str",
        "utf8",
        "categorical",
        "enum",
    }:
        return "string"

    if dtype_name in {"bool", "boolean"}:
        return "boolean"

    if "date" in dtype_name or "time" in dtype_name:
        return "datetime"

    return dtype_name


def compare_dtypes(
    pandas_df: pd.DataFrame,
    polars_df: pl.DataFrame,
) -> dict:
    """Pandas와 Polars의 컬럼별 자료형을 비교한다."""
    dtype_comparison = {}

    for column in pandas_df.columns:
        pandas_dtype = pandas_df[column].dtype
        polars_dtype = polars_df.schema[column]

        pandas_normalized = normalize_dtype(pandas_dtype)
        polars_normalized = normalize_dtype(polars_dtype)

        dtype_comparison[column] = {
            "pandas_dtype": str(pandas_dtype),
            "polars_dtype": str(polars_dtype),
            "normalized_dtype": pandas_normalized,
            "dtype_match": pandas_normalized == polars_normalized,
        }

    return dtype_comparison


def compare_loaders(
    path: Path,
    repeat: int = 5,
    number: int = 3,
) -> dict:
    """
    Pandas와 Polars의 로딩 결과와 성능을 비교한다.

    같은 로컬 파일을 같은 횟수만큼 읽어
    평균 로딩 시간, 메모리, shape, dtype을 비교한다.

    repeat:
        측정 세트의 반복 횟수

    number:
        한 측정 세트 안에서 파일을 불러오는 횟수
    """
    if repeat < 1 or number < 1:
        raise ValueError("repeat와 number는 1 이상이어야 합니다.")

    # 동일한 로딩 함수를 동일한 횟수로 실행한다.
    pandas_total_times = timeit.repeat(
        lambda: load_with_pandas(path),
        repeat=repeat,
        number=number,
    )

    polars_total_times = timeit.repeat(
        lambda: load_with_polars(path),
        repeat=repeat,
        number=number,
    )

    # 각 세트의 전체 시간을 number로 나누어 1회 평균 시간으로 변환한다.
    pandas_load_times = [
        total_time / number
        for total_time in pandas_total_times
    ]

    polars_load_times = [
        total_time / number
        for total_time in polars_total_times
    ]

    # 시간 측정 이후 실제 비교에 사용할 DataFrame을 한 번씩 생성한다.
    pandas_df = load_with_pandas(path)
    polars_df = load_with_polars(path)

    # Pandas는 문자열 객체까지 포함한 메모리 사용량을 계산한다.
    pandas_memory_mb = (
        pandas_df.memory_usage(deep=True).sum()
        / (1024 * 1024)
    )

    # Polars는 DataFrame 내부 버퍼의 예상 크기를 반환한다.
    polars_memory_mb = (
        polars_df.estimated_size()
        / (1024 * 1024)
    )

    dtype_comparison = compare_dtypes(
        pandas_df,
        polars_df,
    )

    dtype_match_count = sum(
        result["dtype_match"]
        for result in dtype_comparison.values()
    )

    columns_match = (
        pandas_df.columns.tolist()
        == polars_df.columns
    )

    shape_match = pandas_df.shape == polars_df.shape

    return {
        "settings": {
            "repeat": repeat,
            "number": number,
            "total_load_count": repeat * number,
        },
        "pandas": {
            "shape": pandas_df.shape,
            "average_load_seconds": mean(pandas_load_times),
            "minimum_load_seconds": min(pandas_load_times),
            "memory_mb": pandas_memory_mb,
        },
        "polars": {
            "shape": polars_df.shape,
            "average_load_seconds": mean(polars_load_times),
            "minimum_load_seconds": min(polars_load_times),
            "memory_mb": polars_memory_mb,
        },
        "comparison": {
            "shape_match": shape_match,
            "columns_match": columns_match,
            "dtype_match_count": dtype_match_count,
            "total_column_count": len(COLUMNS),
            "all_dtypes_match": dtype_match_count == len(COLUMNS),
        },
        "dtypes": dtype_comparison,
    }


def print_loader_comparison(results: dict) -> None:
    """Pandas와 Polars 비교 결과를 터미널에 출력한다."""
    settings = results["settings"]
    pandas_result = results["pandas"]
    polars_result = results["polars"]
    comparison = results["comparison"]

    print("\n[Pandas·Polars 로딩 비교]")
    print(
        f"측정 횟수: "
        f"{settings['repeat']}세트 × "
        f"{settings['number']}회"
    )

    print("\nPandas")
    print(f"- shape: {pandas_result['shape']}")
    print(
        "- 평균 로딩 시간: "
        f"{pandas_result['average_load_seconds']:.6f}초"
    )
    print(
        "- 최소 로딩 시간: "
        f"{pandas_result['minimum_load_seconds']:.6f}초"
    )
    print(
        "- 메모리 사용량: "
        f"{pandas_result['memory_mb']:.2f}MB"
    )

    print("\nPolars")
    print(f"- shape: {polars_result['shape']}")
    print(
        "- 평균 로딩 시간: "
        f"{polars_result['average_load_seconds']:.6f}초"
    )
    print(
        "- 최소 로딩 시간: "
        f"{polars_result['minimum_load_seconds']:.6f}초"
    )
    print(
        "- 메모리 사용량: "
        f"{polars_result['memory_mb']:.2f}MB"
    )

    print("\n비교 결과")
    print(f"- shape 일치: {comparison['shape_match']}")
    print(f"- 컬럼 일치: {comparison['columns_match']}")
    print(
        "- dtype 일치: "
        f"{comparison['dtype_match_count']}/"
        f"{comparison['total_column_count']}"
    )

    print("\n[컬럼별 dtype 비교]")

    for column, dtype_result in results["dtypes"].items():
        print(
            f"- {column}: "
            f"Pandas={dtype_result['pandas_dtype']}, "
            f"Polars={dtype_result['polars_dtype']}, "
            f"공통 분류={dtype_result['normalized_dtype']}, "
            f"일치={dtype_result['dtype_match']}"
        )


if __name__ == "__main__":
    config = load_config()

    data_path = download_data(
        config["data"]["url"],
        config["data"]["raw_path"],
    )

    loader_results = compare_loaders(
        data_path,
        repeat=5,
        number=3,
    )

    print_loader_comparison(loader_results)