"""데이터 로딩과 전처리 기능을 검증하는 테스트."""

from src.data_loader import (
    download_data,
    load_config,
    load_with_pandas,
    load_with_polars,
)
from src.preprocessing import preprocess_data


def test_data_loading_shapes() -> None:
    """Pandas와 Polars가 동일한 크기로 데이터를 읽는지 확인한다."""
    config = load_config()

    raw_path = download_data(
        config["data"]["url"],
        config["data"]["raw_path"],
    )

    pandas_df = load_with_pandas(raw_path)
    polars_df = load_with_polars(raw_path)

    assert pandas_df.shape == (32561, 15)
    assert polars_df.shape == (32561, 15)
    assert pandas_df.columns.tolist() == polars_df.columns


def test_preprocessing_removes_missing_and_duplicates(tmp_path) -> None:
    """전처리 후 결측치와 중복 행이 남지 않는지 확인한다."""
    config = load_config()

    raw_path = download_data(
        config["data"]["url"],
        config["data"]["raw_path"],
    )
    pandas_df = load_with_pandas(raw_path)

    output_path = tmp_path / "adult_processed.csv"

    processed_df, result = preprocess_data(
        pandas_df,
        str(output_path),
    )

    assert processed_df.isna().sum().sum() == 0
    assert processed_df.duplicated().sum() == 0
    assert result["duplicates_before"] == 24
    assert output_path.exists()