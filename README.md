# Adult Census Income End-to-End 분석

Adult Census Income 데이터를 Pandas와 Polars로 불러와 전처리·EDA·시각화·통계 검정·머신러닝을 수행하는 프로젝트입니다.

## 데이터 출처

- 데이터셋: Adult Census Income
- 출처: UCI Machine Learning Repository
- URL: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
- 원본 크기: 32,561행 × 15열

원본 데이터는 GitHub에 포함하지 않으며, 프로그램 실행 시 URL에서 자동으로 내려받습니다.

## 주요 분석

- Pandas·Polars 데이터 로딩 비교
- 결측치 및 중복 행 처리
- Seaborn 나이 분포 시각화
- Plotly 소득별 주당 근무시간 박스플롯
- 수치형 변수 기술통계 및 상관계수
- 소득 그룹별 주당 근무시간 t-test
- RandomForest 분류 Pipeline 학습
- joblib 모델 저장
- `report.md` 자동 생성

## 실행 환경

- Python 3.11
- Pandas
- Polars
- Seaborn
- Plotly
- SciPy
- scikit-learn

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 전체 Pipeline 실행

```bash
python run_pipeline.py
```

실행하면 다음 결과물이 생성됩니다.

- 전처리 데이터: `data/processed/adult_processed.csv`
- Seaborn 차트: `reports/figures/age_distribution.png`
- Plotly 차트: `reports/figures/income_hours_boxplot.html`
- 모델: `models/adult_income_pipeline.joblib`
- 자동 보고서: `reports/report.md`

## 테스트 및 코드 품질

```bash
python -m pytest -v
ruff check .
```

## 주요 결과

- 전처리 후 데이터: 32,537행 × 15열
- 제거한 중복 행: 24개
- 저소득 그룹 평균 근무시간: 38.84시간
- 고소득 그룹 평균 근무시간: 45.47시간
- t-test: 두 그룹의 평균 근무시간 차이가 통계적으로 유의함
- RandomForest Accuracy: 0.8565
- 고소득 그룹 F1-score: 0.6810