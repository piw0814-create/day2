"""전처리된 Adult 데이터의 기본 EDA와 통계 요약을 수행하는 모듈."""

import pandas as pd

from src.data_loader import load_config


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

    # 저소득·고소득 그룹의 데이터 개수를 확인한다.
    target_counts = df[target_column].value_counts()

    # 각 컬럼의 고유값 개수를 확인한다.
    unique_counts = df.nunique()

    return {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "descriptive_stats": descriptive_stats,
        "correlation_matrix": correlation_matrix,
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