"""Adult Census Income 전체 분석 Pipeline 실행 진입점."""

from src.data_loader import (
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

    print("[1/7] 데이터 로딩")
    raw_path = download_data(
        config["data"]["url"],
        config["data"]["raw_path"],
    )
    pandas_df = load_with_pandas(raw_path)
    polars_df = load_with_polars(raw_path)

    print(f"Pandas: {pandas_df.shape}")
    print(f"Polars: {polars_df.shape}")

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

    print("\n[4/7] 시각화 생성")
    visualization_paths = create_visualizations(
        processed_df,
        config["paths"]["figures"],
    )
    print(visualization_paths)

    print("\n[5/7] t-test 수행")
    ttest_result = run_ttest(processed_df)
    print(ttest_result["interpretation"])

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
    print(f"Accuracy: {model_result['accuracy']:.4f}")
    print(f"F1-score: {model_result['f1']:.4f}")

    print("\n[7/7] 보고서 자동 생성")
    report_path = generate_report(
        report_path=config["paths"]["report"],
        pandas_shape=pandas_df.shape,
        polars_shape=polars_df.shape,
        preprocessing_result=preprocessing_result,
        eda_result=eda_result,
        visualization_paths=visualization_paths,
        ttest_result=ttest_result,
        model_result=model_result,
    )

    print(f"\n전체 Pipeline 완료: {report_path}")


if __name__ == "__main__":
    main()