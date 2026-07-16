"""Adult 데이터의 결측치와 중복 행을 처리하는 모듈."""

from pathlib import Path

import pandas as pd

from src.data_loader import download_data, load_config, load_with_pandas


MISSING_COLUMNS = ["workclass", "occupation", "native-country"]


def preprocess_data(
    df: pd.DataFrame,
    output_path: str,
) -> tuple[pd.DataFrame, dict]:
    """결측치와 중복을 처리하고 전처리 결과를 저장한다."""
    cleaned_df = df.copy()

    # 전처리 전 데이터 상태를 기록한다.
    before_shape = cleaned_df.shape
    missing_before = cleaned_df.isna().sum().to_dict()
    duplicates_before = int(cleaned_df.duplicated().sum())

    # 문자열 컬럼의 불필요한 앞뒤 공백을 제거한다.
    string_columns = cleaned_df.select_dtypes(include="str").columns
    cleaned_df[string_columns] = cleaned_df[string_columns].apply(
    lambda column: column.str.strip()
    )   

    # 범주형 결측치는 행을 삭제하지 않고 Unknown 범주로 대체한다.
    cleaned_df[MISSING_COLUMNS] = cleaned_df[MISSING_COLUMNS].fillna("Unknown")
    # 완전히 동일한 중복 행을 제거한다.
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    # 전처리 완료 데이터를 CSV 파일로 저장한다.
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(path, index=False)

    summary = {
        "before_shape": before_shape,
        "after_shape": cleaned_df.shape,
        "missing_before": missing_before,
        "missing_after": cleaned_df.isna().sum().to_dict(),
        "duplicates_before": duplicates_before,
        "duplicates_after": int(cleaned_df.duplicated().sum()),
        "output_path": str(path),
    }

    return cleaned_df, summary


if __name__ == "__main__":
    config = load_config()

    data_path = download_data(
        config["data"]["url"],
        config["data"]["raw_path"],
    )
    pandas_df = load_with_pandas(data_path)

    processed_df, result = preprocess_data(
        pandas_df,
        config["data"]["processed_path"],
    )

    print(f"처리 전 크기: {result['before_shape']}")
    print(f"처리 후 크기: {result['after_shape']}")
    print(f"처리 전 결측치:\n{pandas_df.isna().sum()}")
    print(f"처리 후 결측치:\n{processed_df.isna().sum()}")
    print(f"제거한 중복 행: {result['duplicates_before']}")
    print(f"저장 위치: {result['output_path']}")