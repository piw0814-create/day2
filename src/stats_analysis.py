"""소득 그룹별 주당 근무시간 차이를 t-test로 검정하는 모듈."""

from math import sqrt

import pandas as pd
from scipy.stats import ttest_ind

from src.data_loader import load_config


def interpret_cohens_d(cohens_d: float) -> str:
    """Cohen's d의 절댓값을 기준으로 효과 크기를 해석한다."""
    absolute_d = abs(cohens_d)

    if absolute_d < 0.2:
        return "매우 작은 효과"
    if absolute_d < 0.5:
        return "작은 효과"
    if absolute_d < 0.8:
        return "중간 정도의 효과"

    return "큰 효과"


def run_ttest(
    df: pd.DataFrame,
    target_column: str = "income",
    value_column: str = "hours-per-week",
    alpha: float = 0.05,
) -> dict:
    """두 income 그룹의 평균 주당 근무시간 차이와 효과 크기를 분석한다."""

    low_income = df.loc[
        df[target_column] == "<=50K",
        value_column,
    ].dropna()

    high_income = df.loc[
        df[target_column] == ">50K",
        value_column,
    ].dropna()

    low_count = int(low_income.count())
    high_count = int(high_income.count())

    if low_count < 2 or high_count < 2:
        raise ValueError(
            "t-test와 Cohen's d 계산을 위해 "
            "각 소득 그룹에 최소 2개의 값이 필요합니다."
        )

    low_mean = float(low_income.mean())
    high_mean = float(high_income.mean())

    # 표본 표준편차이므로 ddof=1을 사용한다.
    low_std = float(low_income.std(ddof=1))
    high_std = float(high_income.std(ddof=1))

    # 두 그룹의 분산이 같다고 가정하지 않는 Welch t-test를 사용한다.
    t_statistic, p_value = ttest_ind(
        low_income,
        high_income,
        equal_var=False,
    )

    reject_null = p_value < alpha

    # 두 그룹의 표본 수를 반영한 통합 표준편차를 계산한다.
    pooled_variance = (
        (low_count - 1) * low_std**2
        + (high_count - 1) * high_std**2
    ) / (low_count + high_count - 2)

    pooled_std = sqrt(pooled_variance)

    if pooled_std == 0:
        raise ValueError(
            "통합 표준편차가 0이므로 Cohen's d를 계산할 수 없습니다."
        )

    # 양수이면 고소득 그룹의 평균 근무시간이 더 높다는 의미다.
    cohens_d = (high_mean - low_mean) / pooled_std
    effect_size = interpret_cohens_d(cohens_d)

    if reject_null:
        interpretation = (
            f"p-value가 {alpha}보다 작으므로 귀무가설을 기각한다. "
            "두 소득 그룹의 평균 주당 근무시간에는 "
            "통계적으로 유의한 차이가 있다."
        )
    else:
        interpretation = (
            f"p-value가 {alpha} 이상이므로 귀무가설을 기각하지 못한다. "
            "두 소득 그룹의 평균 주당 근무시간 차이가 "
            "통계적으로 유의하다고 보기 어렵다."
        )

    effect_interpretation = (
        f"Cohen's d는 {cohens_d:.4f}로, "
        f"두 소득 그룹의 평균 근무시간 차이는 {effect_size}로 해석된다."
    )

    return {
        "low_count": low_count,
        "high_count": high_count,
        "low_mean": low_mean,
        "high_mean": high_mean,
        "low_std": low_std,
        "high_std": high_std,
        "pooled_std": pooled_std,
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "reject_null": bool(reject_null),
        "cohens_d": cohens_d,
        "effect_size": effect_size,
        "interpretation": interpretation,
        "effect_interpretation": effect_interpretation,
    }


if __name__ == "__main__":
    config = load_config()
    df = pd.read_csv(config["data"]["processed_path"])

    result = run_ttest(df)

    print(f"저소득 그룹 표본 수: {result['low_count']}")
    print(f"고소득 그룹 표본 수: {result['high_count']}")

    print(
        "저소득 그룹 평균 근무시간: "
        f"{result['low_mean']:.2f}"
    )
    print(
        "고소득 그룹 평균 근무시간: "
        f"{result['high_mean']:.2f}"
    )

    print(
        "저소득 그룹 표준편차: "
        f"{result['low_std']:.4f}"
    )
    print(
        "고소득 그룹 표준편차: "
        f"{result['high_std']:.4f}"
    )
    print(
        "통합 표준편차: "
        f"{result['pooled_std']:.4f}"
    )

    print(f"t-statistic: {result['t_statistic']:.4f}")

    if result["p_value"] == 0:
        print("p-value: < 1e-300")
    else:
        print(f"p-value: {result['p_value']:.6g}")

    print(f"Cohen's d: {result['cohens_d']:.4f}")
    print(f"효과 크기: {result['effect_size']}")

    print(result["interpretation"])
    print(result["effect_interpretation"])