"""분석 결과를 해석하여 Markdown 보고서로 자동 생성하는 모듈."""

from pathlib import Path

import pandas as pd

from src.data_loader import (
    download_data,
    load_config,
    load_with_pandas,
    load_with_polars,
)
from src.eda import analyze_data
from src.ml_pipeline import train_model
from src.preprocessing import preprocess_data
from src.stats_analysis import run_ttest
from src.visualization import create_visualizations


def format_p_value(p_value: float) -> str:
    """p-value를 읽기 쉬운 문자열로 변환한다."""
    if p_value == 0:
        return "< 1e-300"

    if p_value < 0.001:
        return f"{p_value:.3e}"

    return f"{p_value:.6f}"


def interpret_income_distribution(target_counts: pd.Series) -> str:
    """소득 클래스 비율을 계산하여 클래스 불균형 여부를 해석한다."""
    total_count = int(target_counts.sum())
    low_count = int(target_counts.get("<=50K", 0))
    high_count = int(target_counts.get(">50K", 0))

    low_ratio = low_count / total_count
    high_ratio = high_count / total_count

    if abs(low_ratio - high_ratio) >= 0.1:
        balance_text = (
            "두 클래스의 비율 차이가 크므로 클래스 불균형이 존재한다."
        )
    else:
        balance_text = (
            "두 클래스의 비율 차이가 크지 않아 비교적 균형적인 분포이다."
        )

    return (
        f"저소득 그룹은 전체의 {low_ratio:.2%}, "
        f"고소득 그룹은 {high_ratio:.2%}를 차지한다. "
        f"{balance_text}"
    )


def interpret_correlation(correlation_matrix: pd.DataFrame) -> str:
    """자기상관을 제외하고 절댓값이 가장 큰 상관관계를 해석한다."""
    columns = correlation_matrix.columns.tolist()

    strongest_pair: tuple[str, str] | None = None
    strongest_value = 0.0

    # 상관행렬의 대각선과 중복 조합을 제외하고 가장 큰 값을 찾는다.
    for index, first_column in enumerate(columns):
        for second_column in columns[index + 1 :]:
            value = float(
                correlation_matrix.loc[first_column, second_column]
            )

            if pd.isna(value):
                continue

            if abs(value) > abs(strongest_value):
                strongest_value = value
                strongest_pair = (first_column, second_column)

    if strongest_pair is None:
        return "유효한 변수 간 상관계수를 찾지 못했다."

    absolute_value = abs(strongest_value)

    if absolute_value < 0.1:
        strength = "거의 없는"
    elif absolute_value < 0.3:
        strength = "약한"
    elif absolute_value < 0.5:
        strength = "보통 수준의"
    elif absolute_value < 0.7:
        strength = "다소 강한"
    else:
        strength = "강한"

    direction = "양의" if strongest_value > 0 else "음의"
    first_column, second_column = strongest_pair

    return (
        f"자기상관을 제외했을 때 절댓값이 가장 큰 상관관계는 "
        f"`{first_column}`과 `{second_column}` 사이에서 나타났다. "
        f"상관계수는 {strongest_value:.4f}로, "
        f"{strength} {direction} 상관관계이다. "
        "상관관계는 변수 간 연관성을 나타내지만 인과관계를 의미하지는 않는다."
    )


def interpret_hours_comparison(ttest_result: dict) -> str:
    """두 소득 그룹의 평균 근무시간 차이를 실제 결과로 해석한다."""
    low_mean = ttest_result["low_mean"]
    high_mean = ttest_result["high_mean"]
    difference = high_mean - low_mean

    if difference > 0:
        comparison = (
            f"고소득 그룹의 평균 주당 근무시간이 "
            f"저소득 그룹보다 {difference:.2f}시간 높았다."
        )
    elif difference < 0:
        comparison = (
            f"저소득 그룹의 평균 주당 근무시간이 "
            f"고소득 그룹보다 {abs(difference):.2f}시간 높았다."
        )
    else:
        comparison = "두 소득 그룹의 평균 주당 근무시간은 같았다."

    if ttest_result["reject_null"]:
        significance = (
            "t-test 결과 이 평균 차이는 통계적으로 유의했다."
        )
    else:
        significance = (
            "t-test 결과 이 평균 차이는 통계적으로 유의하다고 보기 어려웠다."
        )

    return f"{comparison} {significance}"


def interpret_model_result(model_result: dict) -> str:
    """모델 평가 지표를 비교해 고소득 그룹 예측 특성을 해석한다."""
    precision = model_result["precision"]
    recall = model_result["recall"]

    if recall < precision:
        detail = (
            "고소득 그룹의 Recall이 Precision보다 낮으므로, "
            "고소득으로 예측한 결과의 정확성보다 실제 고소득자를 "
            "놓치는 문제가 상대적으로 더 크게 나타났다."
        )
    elif recall > precision:
        detail = (
            "고소득 그룹의 Recall이 Precision보다 높으므로, "
            "실제 고소득자를 비교적 많이 찾아내지만 일부 저소득자를 "
            "고소득으로 잘못 분류하는 문제가 상대적으로 더 크게 나타났다."
        )
    else:
        detail = (
            "고소득 그룹의 Precision과 Recall이 동일한 수준으로 나타났다."
        )

    return (
        f"모델의 전체 Accuracy는 {model_result['accuracy']:.4f}, "
        f"고소득 그룹 F1-score는 {model_result['f1']:.4f}이다. "
        f"{detail}"
    )


def generate_report(
    report_path: str,
    pandas_shape: tuple[int, int],
    polars_shape: tuple[int, int],
    preprocessing_result: dict,
    eda_result: dict,
    visualization_paths: dict[str, str],
    ttest_result: dict,
    model_result: dict,
) -> Path:
    """실제 분석 결과와 조건별 해석을 포함한 보고서를 생성한다."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 각 분석 결과를 바탕으로 보고서 해석 문장을 동적으로 생성한다.
    income_interpretation = interpret_income_distribution(
        eda_result["target_counts"]
    )
    correlation_interpretation = interpret_correlation(
        eda_result["correlation_matrix"]
    )
    hours_interpretation = interpret_hours_comparison(ttest_result)
    model_interpretation = interpret_model_result(model_result)

    p_value_text = format_p_value(ttest_result["p_value"])

    age_image_name = Path(
        visualization_paths["age_chart"]
    ).name
    boxplot_html_name = Path(
        visualization_paths["income_hours_chart"]
    ).name

    report = f"""# Adult Census Income 분석 보고서

## 1. 프로젝트 개요

Adult Census Income 데이터를 Pandas와 Polars로 불러와 전처리하고,
소득 그룹에 따른 주당 근무시간 차이를 시각화·통계적으로 분석하였다.

이후 수치형 및 범주형 변수로 RandomForest Pipeline을 학습하여
개인의 소득 그룹이 `<=50K`인지 `>50K`인지 예측하였다.

## 2. 데이터 로딩 결과

- Pandas 데이터 크기: `{pandas_shape}`
- Polars 데이터 크기: `{polars_shape}`
- 크기 일치 여부: `{pandas_shape == polars_shape}`

## 3. 데이터 전처리 결과

- 처리 전 크기: `{preprocessing_result["before_shape"]}`
- 처리 후 크기: `{preprocessing_result["after_shape"]}`
- 제거한 중복 행: `{preprocessing_result["duplicates_before"]}`
- 처리 후 중복 행: `{preprocessing_result["duplicates_after"]}`
- 처리 후 결측치 총합: `{sum(preprocessing_result["missing_after"].values())}`

범주형 결측치는 데이터 손실을 줄이기 위해 `Unknown`으로 대체하고,
완전히 동일한 중복 행은 제거하였다.

## 4. 소득 그룹 분포

```text
{eda_result["target_counts"].to_string()}
```

{income_interpretation}

## 5. 수치형 변수 기술통계

```text
{eda_result["descriptive_stats"].round(4).to_string()}
```

기술통계에는 수치형 변수의 평균, 표준편차, 최솟값, 최댓값 및
25%, 50%, 75% 분위수가 포함된다.

## 6. 수치형 변수 상관계수

```text
{eda_result["correlation_matrix"].round(4).to_string()}
```

{correlation_interpretation}

## 7. 시각화 결과

### 나이 분포

![Age Distribution](figures/{age_image_name})

Seaborn 히스토그램과 밀도 곡선을 통해 Adult 데이터의 전체 나이 분포를 확인하였다.

### 소득 그룹별 주당 근무시간

[Plotly 인터랙티브 박스플롯 열기](figures/{boxplot_html_name})

{hours_interpretation}

## 8. 소득 그룹별 주당 근무시간 t-test

- 저소득 그룹 표본 수: `{ttest_result["low_count"]}`
- 고소득 그룹 표본 수: `{ttest_result["high_count"]}`
- 저소득 그룹 평균 근무시간: `{ttest_result["low_mean"]:.2f}`
- 고소득 그룹 평균 근무시간: `{ttest_result["high_mean"]:.2f}`
- 평균 차이: `{ttest_result["high_mean"] - ttest_result["low_mean"]:.2f}`
- t-statistic: `{ttest_result["t_statistic"]:.4f}`
- p-value: `{p_value_text}`
- 유의수준: `{ttest_result["alpha"]}`

### 가설

- 귀무가설: 두 소득 그룹의 평균 주당 근무시간에는 차이가 없다.
- 대립가설: 두 소득 그룹의 평균 주당 근무시간에는 차이가 있다.

{ttest_result["interpretation"]}

## 9. RandomForest 모델 평가

- 학습 데이터 수: `{model_result["train_size"]}`
- 테스트 데이터 수: `{model_result["test_size"]}`
- Accuracy: `{model_result["accuracy"]:.4f}`
- Precision: `{model_result["precision"]:.4f}`
- Recall: `{model_result["recall"]:.4f}`
- F1-score: `{model_result["f1"]:.4f}`

### Confusion Matrix

```text
{model_result["confusion_matrix"]}
```

### Classification Report

```text
{model_result["classification_report"]}
```

- 모델 저장 위치: `{model_result["model_path"]}`

{model_interpretation}

## 10. 결론

{hours_interpretation}

{model_interpretation}

## 11. 한계 및 개선 방향

- 소득 클래스 비율이 다르므로 Accuracy만으로 모델을 평가하기 어렵다.
- 고소득 그룹의 Recall과 F1-score를 함께 확인해야 한다.
- RandomForest 하이퍼파라미터 튜닝을 통해 성능을 개선할 수 있다.
- Logistic Regression 등 다른 분류 모델과 성능을 비교할 수 있다.
- 통계적으로 유의한 차이가 실제로도 중요한 차이인지 효과 크기를 추가로 분석할 수 있다.
"""

    path.write_text(report, encoding="utf-8")
    return path


if __name__ == "__main__":
    config = load_config()

    # 원본 데이터가 없으면 UCI 저장소에서 다운로드한다.
    raw_path = download_data(
        config["data"]["url"],
        config["data"]["raw_path"],
    )

    # 동일한 원본 데이터를 Pandas와 Polars로 각각 불러온다.
    pandas_df = load_with_pandas(raw_path)
    polars_df = load_with_polars(raw_path)

    # 결측치와 중복 행을 처리한다.
    processed_df, preprocessing_result = preprocess_data(
        pandas_df,
        config["data"]["processed_path"],
    )

    # 기술통계, 상관계수 및 소득 그룹 분포를 계산한다.
    eda_result = analyze_data(
        processed_df,
        config["columns"]["numeric"],
        config["columns"]["target"],
    )

    # 정적 및 인터랙티브 시각화를 생성한다.
    visualization_paths = create_visualizations(
        processed_df,
        config["paths"]["figures"],
    )

    # 소득 그룹 간 평균 주당 근무시간 차이를 검정한다.
    ttest_result = run_ttest(processed_df)

    # 전처리와 RandomForest가 연결된 Pipeline을 학습하고 평가한다.
    _, model_result = train_model(
        df=processed_df,
        numeric_columns=config["columns"]["numeric"],
        categorical_columns=config["columns"]["categorical"],
        target_column=config["columns"]["target"],
        model_path=config["paths"]["model"],
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        n_estimators=config["model"]["n_estimators"],
    )

    # 실제 계산 결과와 조건별 해석을 이용해 보고서를 생성한다.
    generated_path = generate_report(
        report_path=config["paths"]["report"],
        pandas_shape=pandas_df.shape,
        polars_shape=polars_df.shape,
        preprocessing_result=preprocessing_result,
        eda_result=eda_result,
        visualization_paths=visualization_paths,
        ttest_result=ttest_result,
        model_result=model_result,
    )

    print(f"보고서 생성 완료: {generated_path}")