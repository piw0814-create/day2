"""소득 그룹별 주당 근무시간 차이를 t-test로 검정하는 모듈."""

import pandas as pd
from scipy.stats import ttest_ind

from src.data_loader import load_config


def run_ttest(
    df: pd.DataFrame,
    target_column: str = "income",
    value_column: str = "hours-per-week",
    alpha: float = 0.05,
) -> dict:
    """두 income 그룹의 평균 주당 근무시간 차이를 검정한다."""

    low_income = df.loc[df[target_column] == "<=50K", value_column]
    high_income = df.loc[df[target_column] == ">50K", value_column]

    # 두 그룹의 분산이 같다고 가정하지 않는 Welch t-test를 사용한다.
    t_statistic, p_value = ttest_ind(
        low_income,
        high_income,
        equal_var=False,
    )

    reject_null = p_value < alpha

    if reject_null:
        interpretation = (
            f"p-value가 {alpha}보다 작으므로 귀무가설을 기각한다. "
            "두 소득 그룹의 평균 주당 근무시간에는 통계적으로 유의한 차이가 있다."
        )
    else:
        interpretation = (
            f"p-value가 {alpha} 이상이므로 귀무가설을 기각하지 못한다. "
            "두 소득 그룹의 평균 주당 근무시간 차이가 통계적으로 유의하다고 보기 어렵다."
        )

    return {
        "low_count": int(low_income.count()),
        "high_count": int(high_income.count()),
        "low_mean": float(low_income.mean()),
        "high_mean": float(high_income.mean()),
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "reject_null": reject_null,
        "interpretation": interpretation,
    }


if __name__ == "__main__":
    config = load_config()
    df = pd.read_csv(config["data"]["processed_path"])

    result = run_ttest(df)

    print(f"저소득 그룹 표본 수: {result['low_count']}")
    print(f"고소득 그룹 표본 수: {result['high_count']}")
    print(f"저소득 그룹 평균 근무시간: {result['low_mean']:.2f}")
    print(f"고소득 그룹 평균 근무시간: {result['high_mean']:.2f}")
    print(f"t-statistic: {result['t_statistic']:.4f}")
    if result["p_value"] == 0:
        print("p-value: < 1e-300")
    else:  
        print(f"p-value: {result['p_value']:.6g}")
    print(result["interpretation"])