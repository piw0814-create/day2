"""Adult Census Income 전체 분석 Pipeline 실행 진입점."""

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
from src.report_generator import generate_report
from src.stats_analysis import run_ttest
from src.visualization import create_visualizations


def main() -> None:
    """데이터 로딩부터 보고서 생성까지 전체 분석 과정을 실행한다."""
    config = load_config()

    print("[1/7] 데이터 로딩 및 Pandas·Polars 비교")

    raw_path = download_data(
        config["data"]["url"],
        config["data"]["raw_path"],
    )

    # 같은 로컬 파일을 같은 횟수로 읽어
    # Pandas와 Polars의 속도·메모리·자료형을 비교한다.
    loader_comparison = compare_loaders(
        raw_path,
        repeat=5,
        number=3,
    )

    pandas_result = loader_comparison["pandas"]
    polars_result = loader_comparison["polars"]
    comparison_result = loader_comparison["comparison"]

    print(
        f"Pandas 평균 로딩 시간: "
        f"{pandas_result['average_load_seconds']:.6f}초"
    )
    print(
        f"Polars 평균 로딩 시간: "
        f"{polars_result['average_load_seconds']:.6f}초"
    )

    print(
        f"Pandas 메모리 사용량: "
        f"{pandas_result['memory_mb']:.2f}MB"
    )
    print(
        f"Polars 메모리 사용량: "
        f"{polars_result['memory_mb']:.2f}MB"
    )

    print(
        f"shape 일치 여부: "
        f"{comparison_result['shape_match']}"
    )
    print(
        f"컬럼 일치 여부: "
        f"{comparison_result['columns_match']}"
    )
    print(
        f"dtype 일치 여부: "
        f"{comparison_result['dtype_match_count']}/"
        f"{comparison_result['total_column_count']}"
    )

    # 구조나 자료형이 다르면 공정한 비교가 아니므로
    # 이후 분석을 진행하지 않고 오류를 발생시킨다.
    if not (
        comparison_result["shape_match"]
        and comparison_result["columns_match"]
        and comparison_result["all_dtypes_match"]
    ):
        raise ValueError(
            "Pandas와 Polars의 로딩 결과가 일치하지 않습니다."
        )

    # 성능 측정과 별도로 이후 분석에서 사용할 데이터를 불러온다.
    pandas_df = load_with_pandas(raw_path)
    polars_df = load_with_polars(raw_path)

    print(f"Pandas shape: {pandas_df.shape}")
    print(f"Polars shape: {polars_df.shape}")

    print("\n[2/7] 데이터 전처리")
    processed_df, preprocessing_result = preprocess_data(
        pandas_df,
        config["data"]["processed_path"],
    )
    print(f"전처리 완료: {processed_df.shape}")

    print("\n[3/7] EDA 및 통계 요약")
    eda_result = analyze_data(
        processed_df,
        config["columns"]["numeric"],
        config["columns"]["target"],
    )
    print(eda_result["target_counts"])

    print("\n[income_binary와 수치형 변수 상관계수]")
    print(eda_result["target_correlations"].round(4))

    print("\n[4/7] 시각화 생성")
    visualization_paths = create_visualizations(
        processed_df,
        config["paths"]["figures"],
    )
    print(visualization_paths)

    print("\n[5/7] t-test 및 효과 크기 분석")
    ttest_result = run_ttest(processed_df)

    print(ttest_result["interpretation"])
    print(
        f"Cohen's d: {ttest_result['cohens_d']:.4f} "
        f"({ttest_result['effect_size']})"
    )

    print("\n[6/7] RandomForest 학습 및 평가")
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

    baseline_result = model_result["baseline"]

    print(f"Accuracy: {model_result['accuracy']:.4f}")
    print(f"F1-score: {model_result['f1']:.4f}")

    print("\n[다수 클래스 베이스라인 비교]")
    print(
        f"베이스라인 예측 클래스: "
        f"{baseline_result['predicted_class']}"
    )
    print(
        f"Baseline Accuracy: "
        f"{baseline_result['accuracy']:.4f}"
    )
    print(
        f"Baseline Recall: "
        f"{baseline_result['recall']:.4f}"
    )
    print(
        f"Baseline F1-score: "
        f"{baseline_result['f1']:.4f}"
    )
    print(
        f"RandomForest Accuracy 개선폭: "
        f"{model_result['accuracy_improvement']:.4f} "
        f"({model_result['accuracy_improvement'] * 100:.2f}%p)"
    )

    print("\n[7/7] 보고서 자동 생성")
    report_path = generate_report(
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
    print(f"\n전체 Pipeline 완료: {report_path}")


if __name__ == "__main__":
    main()