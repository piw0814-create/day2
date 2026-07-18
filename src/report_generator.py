"""분석 결과를 해석하여 Markdown 보고서로 자동 생성하는 모듈."""

from pathlib import Path

import pandas as pd

from src.data_loader import (
    compare_loaders,
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


def interpret_loader_comparison(loader_comparison: dict) -> str:
    """Pandas와 Polars의 로딩 시간 및 메모리 사용량을 해석한다."""
    pandas_result = loader_comparison["pandas"]
    polars_result = loader_comparison["polars"]

    pandas_time = pandas_result["average_load_seconds"]
    polars_time = polars_result["average_load_seconds"]

    pandas_memory = pandas_result["memory_mb"]
    polars_memory = polars_result["memory_mb"]

    if pandas_time > polars_time:
        time_ratio = polars_time / pandas_time
        time_text = (
            "이번 실행 환경에서는 Polars의 평균 로딩 시간이 "
            f"Pandas의 {time_ratio:.2%} 수준으로 측정되었다."
        )
    elif pandas_time < polars_time:
        time_ratio = pandas_time / polars_time
        time_text = (
            "이번 실행 환경에서는 Pandas의 평균 로딩 시간이 "
            f"Polars의 {time_ratio:.2%} 수준으로 측정되었다."
        )
    else:
        time_text = (
            "이번 실행 환경에서는 Pandas와 Polars의 "
            "평균 로딩 시간이 동일하게 측정되었다."
        )

    if pandas_memory > polars_memory:
        memory_ratio = polars_memory / pandas_memory
        memory_text = (
            "Polars의 메모리 사용량은 "
            f"Pandas의 {memory_ratio:.2%} 수준으로 측정되었다."
        )
    elif pandas_memory < polars_memory:
        memory_ratio = pandas_memory / polars_memory
        memory_text = (
            "Pandas의 메모리 사용량은 "
            f"Polars의 {memory_ratio:.2%} 수준으로 측정되었다."
        )
    else:
        memory_text = (
            "Pandas와 Polars의 메모리 사용량은 동일하게 측정되었다."
        )

    return (
        f"{time_text} {memory_text} "
        "실행 시간은 컴퓨터 상태와 실행 시점에 따라 달라질 수 있으므로, "
        "이번 실행 환경에서 측정된 결과로 해석해야 한다."
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
        "상관관계는 변수 간 연관성을 나타내지만 "
        "인과관계를 의미하지는 않는다."
    )


def interpret_target_correlations(
    target_correlations: pd.Series,
) -> str:
    """수치형 변수와 소득 그룹 사이의 상관관계를 해석한다."""
    if target_correlations.empty:
        return "소득 그룹과의 상관관계를 계산할 수 없었다."

    strongest_column = target_correlations.abs().idxmax()
    strongest_value = float(
        target_correlations.loc[strongest_column]
    )

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

    if strongest_value > 0:
        direction = "양의"
    elif strongest_value < 0:
        direction = "음의"
    else:
        direction = "방향이 없는"

    near_zero_columns = target_correlations[
        target_correlations.abs() < 0.1
    ].index.tolist()

    if near_zero_columns:
        near_zero_names = ", ".join(
            f"`{column}`"
            for column in near_zero_columns
        )
        near_zero_text = (
            f" 반면 {near_zero_names}은 소득 그룹과의 "
            "선형 관계가 거의 나타나지 않았다."
        )
    else:
        near_zero_text = ""

    return (
        f"수치형 변수 중 `{strongest_column}`이 소득 그룹과 "
        f"가장 큰 상관관계를 보였다. "
        f"상관계수는 {strongest_value:.4f}로, "
        f"{strength} {direction} 상관관계이다."
        f"{near_zero_text} "
        "이 결과는 변수와 소득 그룹 사이의 연관성을 나타내며, "
        "해당 변수가 소득의 직접적인 원인이라는 의미는 아니다."
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

def interpret_baseline_comparison(model_result: dict) -> str:
    """다수 클래스 베이스라인과 RandomForest 성능을 비교한다."""
    baseline = model_result["baseline"]
    improvement = model_result["accuracy_improvement"]

    if improvement > 0:
        accuracy_text = (
            f"RandomForest의 Accuracy는 베이스라인보다 "
            f"{improvement * 100:.2f}%p 높았다."
        )
    elif improvement < 0:
        accuracy_text = (
            f"RandomForest의 Accuracy는 베이스라인보다 "
            f"{abs(improvement) * 100:.2f}%p 낮았다."
        )
    else:
        accuracy_text = (
            "RandomForest와 베이스라인의 Accuracy는 동일했다."
        )

    if baseline["recall"] == 0 and baseline["f1"] == 0:
        minority_text = (
            "베이스라인은 모든 데이터를 다수 클래스인 "
            f"`{baseline['predicted_class']}`로 예측했기 때문에 "
            "고소득 그룹을 한 명도 찾아내지 못했다."
        )
    else:
        minority_text = (
            "베이스라인도 일부 고소득 그룹을 구분하였다."
        )

    return (
        f"다수 클래스 베이스라인의 Accuracy는 "
        f"{baseline['accuracy']:.4f}이고, "
        f"RandomForest의 Accuracy는 "
        f"{model_result['accuracy']:.4f}이다. "
        f"{accuracy_text} {minority_text}"
    )

def generate_report(
    report_path: str,
    pandas_shape: tuple[int, int],
    polars_shape: tuple[int, int],
    loader_comparison: dict,
    preprocessing_result: dict,
    eda_result: dict,
    visualization_paths: dict[str, str],
    ttest_result: dict,
    model_result: dict,
) -> Path:
    """실제 분석 결과와 조건별 해석을 포함한 보고서를 생성한다."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    loader_interpretation = interpret_loader_comparison(
        loader_comparison
    )

    loader_settings = loader_comparison["settings"]
    pandas_loader = loader_comparison["pandas"]
    polars_loader = loader_comparison["polars"]
    loader_check = loader_comparison["comparison"]

    income_interpretation = interpret_income_distribution(
        eda_result["target_counts"]
    )
    correlation_interpretation = interpret_correlation(
        eda_result["correlation_matrix"]
    )
    target_correlation_interpretation = (
        interpret_target_correlations(
            eda_result["target_correlations"]
        )
    )

    hours_interpretation = interpret_hours_comparison(ttest_result)
    model_interpretation = interpret_model_result(model_result)
    baseline_interpretation = interpret_baseline_comparison(
        model_result
    )

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

## 2. Pandas·Polars 로딩 비교

동일한 로컬 원본 파일을 대상으로 Pandas와 Polars를 각각
`{loader_settings["repeat"]}`세트 × `{loader_settings["number"]}`회 실행하였다.

다운로드 시간은 제외하고 파일을 DataFrame으로 불러오는 시간만 측정하였다.

| 비교 항목 | Pandas | Polars |
|---|---:|---:|
| 데이터 크기 | `{pandas_loader["shape"]}` | `{polars_loader["shape"]}` |
| 평균 로딩 시간 | `{pandas_loader["average_load_seconds"]:.6f}초` | `{polars_loader["average_load_seconds"]:.6f}초` |
| 최소 로딩 시간 | `{pandas_loader["minimum_load_seconds"]:.6f}초` | `{polars_loader["minimum_load_seconds"]:.6f}초` |
| 메모리 사용량 | `{pandas_loader["memory_mb"]:.2f}MB` | `{polars_loader["memory_mb"]:.2f}MB` |

### 로딩 결과 검증

- shape 일치 여부: `{loader_check["shape_match"]}`
- 컬럼 이름 및 순서 일치 여부: `{loader_check["columns_match"]}`
- dtype 일치: `{loader_check["dtype_match_count"]}/{loader_check["total_column_count"]}`
- 모든 dtype 일치 여부: `{loader_check["all_dtypes_match"]}`

{loader_interpretation}

Pandas와 Polars는 같은 의미의 자료형을 서로 다른 이름으로 표현할 수 있다.

예를 들어 Pandas의 `int64`와 Polars의 `Int64`,
Pandas의 문자열 자료형과 Polars의 `String`을 공통 범주로 변환하여 비교하였다.

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

## 7. 소득 그룹과 수치형 변수의 상관관계

문자열로 구성된 소득 그룹을 상관계수 계산을 위해 다음과 같이 변환하였다.

- `<=50K`: `0`
- `>50K`: `1`

```text
{eda_result["target_correlations"].round(4).to_string()}
```

{target_correlation_interpretation}

양의 상관계수는 해당 수치형 변수의 값이 높을수록
`>50K` 그룹과 연결되는 경향이 있음을 의미한다.

이 결과는 각 수치형 변수와 소득 그룹 사이의 선형 관계를 나타내며,
변수가 소득의 직접적인 원인이라는 의미는 아니다.

## 8. 시각화 결과

### 나이 분포

![Age Distribution](figures/{age_image_name})

Seaborn 히스토그램과 밀도 곡선을 통해
Adult 데이터의 전체 나이 분포를 확인하였다.

### 소득 그룹별 주당 근무시간

[Plotly 인터랙티브 박스플롯 열기](figures/{boxplot_html_name})

{hours_interpretation}

## 9. 소득 그룹별 주당 근무시간 t-test 및 효과 크기

- 저소득 그룹 표본 수: `{ttest_result["low_count"]}`
- 고소득 그룹 표본 수: `{ttest_result["high_count"]}`
- 저소득 그룹 평균 근무시간: `{ttest_result["low_mean"]:.2f}`
- 고소득 그룹 평균 근무시간: `{ttest_result["high_mean"]:.2f}`
- 평균 차이: `{ttest_result["high_mean"] - ttest_result["low_mean"]:.2f}`
- 저소득 그룹 표준편차: `{ttest_result["low_std"]:.4f}`
- 고소득 그룹 표준편차: `{ttest_result["high_std"]:.4f}`
- 통합 표준편차: `{ttest_result["pooled_std"]:.4f}`
- Cohen's d: `{ttest_result["cohens_d"]:.4f}`
- 효과 크기: `{ttest_result["effect_size"]}`
- t-statistic: `{ttest_result["t_statistic"]:.4f}`
- p-value: `{p_value_text}`
- 유의수준: `{ttest_result["alpha"]}`

### 가설

- 귀무가설: 두 소득 그룹의 평균 주당 근무시간에는 차이가 없다.
- 대립가설: 두 소득 그룹의 평균 주당 근무시간에는 차이가 있다.

{ttest_result["interpretation"]}

{ttest_result["effect_interpretation"]}

## 10. RandomForest 모델 평가

- 학습 데이터 수: `{model_result["train_size"]}`
- 테스트 데이터 수: `{model_result["test_size"]}`
- Accuracy: `{model_result["accuracy"]:.4f}`
- Precision: `{model_result["precision"]:.4f}`
- Recall: `{model_result["recall"]:.4f}`
- F1-score: `{model_result["f1"]:.4f}`

### 다수 클래스 베이스라인 비교

다수 클래스 베이스라인은 학습 데이터에서 가장 많은 클래스로
모든 테스트 데이터를 예측하는 단순 기준 모델이다.

| 평가 지표 | 다수 클래스 베이스라인 | RandomForest |
|---|---:|---:|
| Accuracy | `{model_result["baseline"]["accuracy"]:.4f}` | `{model_result["accuracy"]:.4f}` |
| Precision | `{model_result["baseline"]["precision"]:.4f}` | `{model_result["precision"]:.4f}` |
| Recall | `{model_result["baseline"]["recall"]:.4f}` | `{model_result["recall"]:.4f}` |
| F1-score | `{model_result["baseline"]["f1"]:.4f}` | `{model_result["f1"]:.4f}` |

- 베이스라인 예측 클래스: `{model_result["baseline"]["predicted_class"]}`
- RandomForest Accuracy 개선폭: `{model_result["accuracy_improvement"]:.4f}`
- 퍼센트포인트 기준 개선폭: `{model_result["accuracy_improvement"] * 100:.2f}%p`

{baseline_interpretation}

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

## 11. 결론

{target_correlation_interpretation}

{hours_interpretation}

{ttest_result["effect_interpretation"]}

{model_interpretation}

{baseline_interpretation}

## 12. 한계 및 개선 방향

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

    raw_path = download_data(
        config["data"]["url"],
        config["data"]["raw_path"],
    )

    loader_comparison = compare_loaders(
        raw_path,
        repeat=5,
        number=3,
    )

    pandas_df = load_with_pandas(raw_path)
    polars_df = load_with_polars(raw_path)

    processed_df, preprocessing_result = preprocess_data(
        pandas_df,
        config["data"]["processed_path"],
    )

    eda_result = analyze_data(
        processed_df,
        config["columns"]["numeric"],
        config["columns"]["target"],
    )

    visualization_paths = create_visualizations(
        processed_df,
        config["paths"]["figures"],
    )

    ttest_result = run_ttest(processed_df)

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

    generated_path = generate_report(
        report_path=config["paths"]["report"],
        pandas_shape=pandas_df.shape,
        polars_shape=polars_df.shape,
        loader_comparison=loader_comparison,
        preprocessing_result=preprocessing_result,
        eda_result=eda_result,
        visualization_paths=visualization_paths,
        ttest_result=ttest_result,
        model_result=model_result,
    )

    print(f"보고서 생성 완료: {generated_path}")