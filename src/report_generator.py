"""분석 결과를 Markdown 보고서로 자동 생성하는 모듈."""

from pathlib import Path

from src.data_loader import (
    download_data,
    load_config,
    load_with_pandas,
    load_with_polars,
)
from src.eda import analyze_data
from src.preprocessing import preprocess_data
from src.visualization import create_visualizations
from src.stats_analysis import run_ttest
from src.ml_pipeline import train_model

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
    """데이터 로딩, 전처리, EDA 결과가 포함된 보고서를 생성한다."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    p_value_text = (
    "< 1e-300"
    if ttest_result["p_value"] == 0
    else f"{ttest_result['p_value']:.6g}"
    )

    report = f"""# Adult Census Income 분석 보고서

## 1. 데이터 로딩 결과

- Pandas 데이터 크기: `{pandas_shape}`
- Polars 데이터 크기: `{polars_shape}`
- 크기 일치 여부: `{pandas_shape == polars_shape}`

## 2. 데이터 전처리 결과

- 처리 전 크기: `{preprocessing_result["before_shape"]}`
- 처리 후 크기: `{preprocessing_result["after_shape"]}`
- 제거한 중복 행: `{preprocessing_result["duplicates_before"]}`
- 처리 후 중복 행: `{preprocessing_result["duplicates_after"]}`
- 처리 후 결측치 총합: `{sum(preprocessing_result["missing_after"].values())}`

범주형 결측치는 `Unknown`으로 대체하고,
완전히 동일한 중복 행은 제거하였다.

## 3. 소득 그룹 분포

```text
{eda_result["target_counts"].to_string()}
```

## 4. 수치형 변수 기술통계

```text
{eda_result["descriptive_stats"].round(4).to_string()}
```
## 5. 수치형 변수 상관계수

```text
{eda_result["correlation_matrix"].round(4).to_string()}
```

수치형 변수 간 상관관계는 전반적으로 강하지 않았고,
`education-num`과 `hours-per-week`가 약한 양의 상관관계를 보였다.

## 6. 시각화 결과

### 나이 분포

![Age Distribution](figures/age_distribution.png)

### 소득 그룹별 주당 근무시간

[Plotly 인터랙티브 박스플롯 열기](figures/income_hours_boxplot.html)

고소득 그룹은 저소득 그룹보다 주당 근무시간이 전반적으로 높은 경향을 보였다.

## 7. 소득 그룹별 주당 근무시간 t-test

- 저소득 그룹 표본 수: `{ttest_result["low_count"]}`
- 고소득 그룹 표본 수: `{ttest_result["high_count"]}`
- 저소득 그룹 평균 근무시간: `{ttest_result["low_mean"]:.2f}`
- 고소득 그룹 평균 근무시간: `{ttest_result["high_mean"]:.2f}`
- t-statistic: `{ttest_result["t_statistic"]:.4f}`
- p-value: `{p_value_text}`
- 유의수준: `{ttest_result["alpha"]}`

{ttest_result["interpretation"]}

## 8. RandomForest 모델 평가

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

## 9. 결론

고소득 그룹의 평균 주당 근무시간은 `{ttest_result["high_mean"]:.2f}`시간으로,
저소득 그룹의 `{ttest_result["low_mean"]:.2f}`시간보다 높았다.

t-test 결과 p-value가 유의수준보다 작아,
두 소득 그룹의 평균 근무시간 차이는 통계적으로 유의했다.

RandomForest 베이스라인 모델은 Accuracy `{model_result["accuracy"]:.4f}`,
고소득 그룹 F1-score `{model_result["f1"]:.4f}`를 기록했다.

## 10. 한계 및 개선 방향

- 소득 클래스의 개수 차이로 인해 클래스 불균형이 존재한다.
- RandomForest 하이퍼파라미터 튜닝을 추가할 수 있다.
- Logistic Regression 등 다른 모델과 성능을 비교할 수 있다.
- Accuracy뿐 아니라 고소득 그룹의 Recall과 F1-score도 함께 고려해야 한다.

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

    # 전처리된 데이터의 기본 EDA 결과를 계산한다.
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

    # 실제 분석 결과를 이용해 Markdown 보고서를 자동 생성한다.
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