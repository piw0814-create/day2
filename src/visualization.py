"""Adult 데이터의 정적·인터랙티브 시각화를 생성하는 모듈."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from src.data_loader import load_config


def create_visualizations(
    df: pd.DataFrame,
    figures_dir: str,
) -> dict[str, str]:
    """Seaborn과 Plotly 차트를 생성하고 파일로 저장한다."""
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    age_chart_path = output_dir / "age_distribution.png"
    income_hours_path = output_dir / "income_hours_boxplot.html"

    # 전체 데이터의 나이 분포를 Seaborn 정적 차트로 표현한다.
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x="age",
        bins=30,
        kde=True,
    )

    plt.title("Adult Census Income - Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(age_chart_path, dpi=300)
    plt.close()

    # 소득 그룹별 주당 근무시간 분포를 Plotly 박스플롯으로 비교한다.
    boxplot = px.box(
        df,
        x="income",
        y="hours-per-week",
        category_orders={
            "income": ["<=50K", ">50K"],
        },
        title="Hours per Week by Income Group",
        labels={
            "income": "Income Group",
            "hours-per-week": "Hours per Week",
        },
    )

    # 고정된 div_id를 사용해 실행할 때마다 HTML 내용이 달라지는 것을 방지한다.
    boxplot.write_html(
        income_hours_path,
        div_id="income-hours-boxplot",
    )

    # 생성된 시각화 파일의 경로를 호출한 코드에 반환한다.
    return {
        "age_chart": str(age_chart_path),
        "income_hours_chart": str(income_hours_path),
    }


if __name__ == "__main__":
    config = load_config()
    processed_df = pd.read_csv(
        config["data"]["processed_path"],
    )

    saved_paths = create_visualizations(
        processed_df,
        config["paths"]["figures"],
    )

    print(f"Seaborn 차트 저장: {saved_paths['age_chart']}")
    print(f"Plotly 차트 저장: {saved_paths['income_hours_chart']}")