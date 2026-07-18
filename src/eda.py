"""전처리된 Adult 데이터의 기본 EDA와 통계 요약을 수행하는 모듈."""

import pandas as pd

from src.data_loader import load_config


INCOME_BINARY_MAP = {
    "<=50K": 0,
    ">50K": 1,
}


def calculate_target_correlations(
    df: pd.DataFrame,
    numeric_columns: list[str],
    target_column: str,
) -> pd.Series:
    """
    소득 그룹을 0과 1로 변환하고 수치형 변수와의 상관계수를 계산한다.

    <=50K는 0, >50K는 1로 변환한다.
    양의 상관계수는 변수값이 높을수록 고소득 그룹과 연결되는
    경향이 있음을 의미한다.
    """
    if df[target_column].isna().any():
        raise ValueError(
            f"{target_column} 컬럼에 결측치가 있어 "
            "income_binary를 생성할 수 없습니다."
        )

    target_values = set(df[target_column].unique())
    unexpected_values = target_values - set(INCOME_BINARY_MAP)

    if unexpected_values:
        raise ValueError(
            "예상하지 못한 income 값이 있습니다: "
            f"{sorted(unexpected_values)}"
        )

    # 원본 DataFrame을 변경하지 않기 위해 분석용 복사본을 만든다.
    correlation_df = df[numeric_columns].copy()

    correlation_df["income_binary"] = df[target_column].map(
        INCOME_BINARY_MAP
    )

    # income_binary와 각 수치형 변수의 피어슨 상관계수를 계산한다.
    target_correlations = (
        correlation_df.corr()["income_binary"]
        .drop("income_binary")
    )

    # 양수·음수와 관계없이 연관성이 큰 변수부터 확인할 수 있도록
    # 상관계수 절댓값을 기준으로 정렬하되 원래 부호는 유지한다.
    sorted_index = (
        target_correlations.abs()
        .sort_values(ascending=False)
        .index
    )

    return target_correlations.reindex(sorted_index)


def analyze_data(
    df: pd.DataFrame,
    numeric_columns: list[str],
    target_column: str,
) -> dict:
    """데이터 구조, 기술통계, 상관계수, 타깃 분포를 계산한다."""

    # 수치형 6개 변수의 평균, 표준편차, 분위수 등을 계산한다.
    descriptive_stats = df[numeric_columns].describe()

    # 수치형 변수 사이의 피어슨 상관계수를 계산한다.
    correlation_matrix = df[numeric_columns].corr()

    # 각 수치형 변수와 소득 그룹 사이의 상관계수를 계산한다.
    target_correlations = calculate_target_correlations(
        df,
        numeric_columns,
        target_column,
    )

    # 저소득·고소득 그룹의 데이터 개수를 확인한다.
    target_counts = df[target_column].value_counts()

    # 각 컬럼의 고유값 개수를 확인한다.
    unique_counts = df.nunique()

    return {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "descriptive_stats": descriptive_stats,
        "correlation_matrix": correlation_matrix,
        "target_correlations": target_correlations,
        "target_counts": target_counts,
        "unique_counts": unique_counts,
    }


if __name__ == "__main__":
    config = load_config()

    # 전처리가 완료된 CSV 데이터를 불러온다.
    df = pd.read_csv(config["data"]["processed_path"])

    results = analyze_data(
        df,
        config["columns"]["numeric"],
        config["columns"]["target"],
    )

    print(f"데이터 크기: {results['shape']}")

    print("\n[컬럼별 자료형]")
    print(results["dtypes"])

    print("\n[income 그룹 분포]")
    print(results["target_counts"])

    print("\n[컬럼별 고유값 개수]")
    print(results["unique_counts"])

    print("\n[수치형 변수 기술통계]")
    print(results["descriptive_stats"])

    print("\n[수치형 변수 상관계수]")
    print(results["correlation_matrix"])

    print("\n[income_binary와 수치형 변수의 상관계수]")
    print(results["target_correlations"])