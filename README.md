# Adult Census Income End-to-End 분석

Adult Census Income 데이터를 이용하여 데이터 로딩부터 전처리, 탐색적 데이터 분석,
시각화, 통계 검정, 머신러닝 학습, 보고서 생성까지 하나의 Pipeline으로 구현한 프로젝트입니다.

Pandas와 Polars로 동일한 원본 데이터를 불러와 데이터 크기와 컬럼 구조를 비교하고,
이후 Pandas DataFrame을 기준으로 전처리·EDA·통계 분석·머신러닝을 수행합니다.

RandomForest 모델을 이용하여 개인의 소득 그룹이
`<=50K`인지 `>50K`인지 예측합니다.

---

## 프로젝트 목표

이 프로젝트의 목표는 단순히 모델 하나를 학습하는 것이 아니라,
데이터 분석의 전체 과정을 하나의 실행 흐름으로 연결하는 것입니다.

```text
원본 데이터 다운로드
→ Pandas·Polars 데이터 로딩 및 구조 비교
→ 결측치·중복 행 전처리
→ 기술통계·상관관계 분석
→ Seaborn·Plotly 시각화
→ 소득 그룹 간 t-test
→ RandomForest 학습 및 평가
→ 모델·시각화·보고서 저장
```

---

## 데이터 출처

- 데이터셋: Adult Census Income
- 출처: UCI Machine Learning Repository
- URL: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
- 원본 크기: 32,561행 × 15열
- 예측 대상: `income`
- 클래스:
  - `<=50K`
  - `>50K`

원본 데이터는 GitHub에 포함하지 않습니다.

`python run_pipeline.py`를 실행하면 UCI 저장소에서 원본 데이터를 자동으로 내려받아
`data/raw/adult.data`에 저장합니다.

---

## 프로젝트 구조

```text
DAY2/
├── README.md
├── requirements.txt
├── config.yaml
├── run_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   ├── report.md
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── visualization.py
│   ├── stats_analysis.py
│   ├── ml_pipeline.py
│   └── report_generator.py
├── tests/
│   └── test_data_pipeline.py
├── notebooks/
└── presentation/
```

---

## 파일별 역할

| 파일 | 역할 |
|---|---|
| `run_pipeline.py` | 전체 분석 과정을 정해진 순서대로 실행 |
| `config.yaml` | 데이터 경로, 컬럼 목록, 모델 설정 관리 |
| `src/data_loader.py` | 원본 데이터 다운로드 및 Pandas·Polars 로딩 |
| `src/preprocessing.py` | 결측치와 중복 행 처리 |
| `src/eda.py` | 기술통계, 상관계수, 소득 분포 분석 |
| `src/visualization.py` | Seaborn과 Plotly 시각화 생성 |
| `src/stats_analysis.py` | 소득 그룹별 주당 근무시간 Welch t-test |
| `src/ml_pipeline.py` | 전처리와 RandomForest 학습 및 평가 |
| `src/report_generator.py` | 실제 분석 결과를 이용한 Markdown 보고서 생성 |
| `tests/test_data_pipeline.py` | 데이터 로딩과 전처리 결과 검증 |

---

## 주요 분석

- Pandas와 Polars의 데이터 크기 및 컬럼 구조 비교
- 범주형 결측치를 `Unknown`으로 처리
- 완전히 동일한 중복 행 제거
- 수치형 변수 기술통계 계산
- 수치형 변수 간 상관계수 분석
- 소득 클래스 분포 확인
- Seaborn을 이용한 나이 분포 시각화
- Plotly를 이용한 소득별 주당 근무시간 박스플롯
- 소득 그룹별 주당 근무시간 Welch t-test
- RandomForest 분류 Pipeline 학습
- Accuracy, Precision, Recall, F1-score 및 혼동행렬 평가
- 전처리와 모델이 연결된 전체 Pipeline을 joblib로 저장
- 실제 실행 결과에 따라 `report.md` 자동 생성

---

## 개발 과정

### 1. 데이터 구조 확인

UCI Adult 데이터에는 별도의 컬럼명이 포함되어 있지 않기 때문에,
15개 컬럼명을 코드에서 직접 지정했습니다.

원본 데이터의 범주형 결측치는 `?`로 표현되어 있어,
데이터를 불러올 때 해당 값을 결측치로 인식하도록 설정했습니다.

### 2. Pandas와 Polars 로딩

동일한 원본 데이터를 Pandas와 Polars로 각각 불러온 뒤,
다음 내용을 확인했습니다.

- 행과 열의 개수
- 컬럼 이름과 순서
- 문자열 앞뒤 공백 처리
- 비어 있는 행 포함 여부

두 라이브러리의 최종 데이터 크기가 모두 `(32561, 15)`인지 확인하고,
컬럼 구조가 동일한지 검증했습니다.

본 프로젝트에서는 두 라이브러리의 로딩 결과가 동일한지를 비교했습니다.

이후 전처리, EDA, 통계 분석 및 머신러닝은
scikit-learn과의 연동을 고려하여 Pandas DataFrame을 기준으로 수행했습니다.

Pandas와 Polars의 로딩 시간 및 메모리 사용량 비교는 현재 범위에 포함하지 않았으며,
추후 개선 항목으로 확장할 수 있습니다.

### 3. 데이터 전처리

범주형 변수의 결측치는 해당 행을 삭제하지 않고 `Unknown`으로 대체했습니다.

이 방법은 결측치가 존재한다는 정보를 유지하면서
데이터 손실을 줄이기 위해 선택했습니다.

완전히 동일한 중복 행 24개는 제거하여,
최종적으로 32,537개 행을 분석에 사용했습니다.

### 4. 탐색적 데이터 분석

수치형 변수에 대해 다음 정보를 계산했습니다.

- 평균
- 표준편차
- 최솟값과 최댓값
- 25%, 50%, 75% 사분위수
- 변수 간 상관계수
- 소득 클래스 분포

소득 클래스는 `<=50K`가 75.91%, `>50K`가 24.09%로 나타나
클래스 불균형이 있음을 확인했습니다.

### 5. 시각화

Seaborn으로 전체 나이 분포를 정적 이미지로 저장하고,
Plotly로 소득 그룹별 주당 근무시간을 비교하는
인터랙티브 박스플롯을 생성했습니다.

- Seaborn: 나이 분포 히스토그램 및 밀도 곡선
- Plotly: 소득 그룹별 주당 근무시간 박스플롯

### 6. 통계 검정

두 소득 그룹의 평균 주당 근무시간 차이를 확인하기 위해
Welch의 독립표본 t-test를 사용했습니다.

두 그룹의 분산이 같다고 가정하지 않도록
`equal_var=False`를 적용했습니다.

- 귀무가설: 두 소득 그룹의 평균 주당 근무시간에는 차이가 없다.
- 대립가설: 두 소득 그룹의 평균 주당 근무시간에는 차이가 있다.

### 7. 머신러닝 Pipeline

수치형 변수와 범주형 변수에 서로 다른 전처리를 적용했습니다.

- 수치형 변수: 중앙값으로 결측치 대체
- 범주형 변수: 최빈값으로 결측치 대체
- 범주형 변수: One-Hot Encoding 적용
- 분류 모델: `RandomForestClassifier`

전처리 과정과 RandomForest 모델을 하나의 scikit-learn Pipeline으로 연결하여,
학습과 예측 과정에서 동일한 전처리가 적용되도록 구성했습니다.

학습이 완료된 전체 Pipeline은 joblib 파일로 저장합니다.

### 8. 자동 보고서 생성

각 분석 단계에서 나온 결과를 `report_generator.py`에 전달하여
`reports/report.md`가 자동으로 생성되도록 구현했습니다.

보고서의 제목과 분석 방법 설명은 미리 작성하지만,
다음 내용은 실제 실행 결과를 조건문으로 판단하여 생성합니다.

- 소득 클래스 불균형 여부
- 절댓값이 가장 큰 변수 간 상관관계
- 상관관계의 방향과 강도
- 어느 소득 그룹의 평균 근무시간이 더 높은지
- 평균 차이가 통계적으로 유의한지
- Precision과 Recall 중 어느 지표가 더 낮은지
- 모델 평가 결과에 대한 해석

따라서 데이터나 모델의 결과가 달라지면
보고서에 작성되는 해석도 함께 변경됩니다.

---

## 설계 의도

전체 분석 코드를 하나의 파일에 작성하지 않고,
데이터 로딩, 전처리, EDA, 시각화, 통계, 모델링, 보고서 생성을
각각의 모듈로 분리했습니다.

기능별로 파일을 나누면 코드가 길어져도 각 부분의 역할을 쉽게 파악할 수 있고,
오류가 발생했을 때 수정할 파일도 빠르게 찾을 수 있습니다.

데이터 경로, 컬럼 목록, 모델 설정은 `config.yaml`에서 관리하여
코드 내부에 설정값이 반복되지 않도록 했습니다.

또한 `run_pipeline.py`는 각 기능을 직접 구현하지 않고,
각 모듈의 함수를 정해진 순서대로 호출하는 역할만 담당하도록 구성했습니다.

---

## 실행 환경

- Python 3.11
- Pandas
- Polars
- Seaborn
- Matplotlib
- Plotly
- SciPy
- scikit-learn
- joblib
- PyYAML
- pytest
- Ruff

---

## 설치

macOS, Linux, WSL 환경에서 다음 명령을 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 전체 Pipeline 실행

```bash
python run_pipeline.py
```

Pipeline은 다음 7단계로 실행됩니다.

```text
[1/7] 데이터 로딩
[2/7] 데이터 전처리
[3/7] EDA 및 통계 요약
[4/7] 시각화 생성
[5/7] t-test 수행
[6/7] RandomForest 학습 및 평가
[7/7] 보고서 자동 생성
```

---

## 생성 결과물

- 원본 데이터: `data/raw/adult.data`
- 전처리 데이터: `data/processed/adult_processed.csv`
- Seaborn 차트: `reports/figures/age_distribution.png`
- Plotly 차트: `reports/figures/income_hours_boxplot.html`
- 학습된 Pipeline: `models/adult_income_pipeline.joblib`
- 자동 분석 보고서: `reports/report.md`

---

## 주요 결과

### 데이터 전처리

- 원본 데이터: 32,561행 × 15열
- 전처리 후 데이터: 32,537행 × 15열
- 제거한 중복 행: 24개
- 처리 후 중복 행: 0개
- 처리 후 결측치: 0개

### 소득 클래스 분포

- `<=50K`: 24,698개, 75.91%
- `>50K`: 7,839개, 24.09%

저소득 그룹이 고소득 그룹보다 약 3배 많으므로,
모델 평가 시 클래스 불균형을 고려해야 합니다.

### 소득 그룹별 주당 근무시간

- 저소득 그룹 평균: 38.84시간
- 고소득 그룹 평균: 45.47시간
- 평균 차이: 6.63시간
- t-statistic: -45.0950
- p-value: `< 1e-300`
- 유의수준: 0.05

p-value가 유의수준 0.05보다 작으므로 귀무가설을 기각했습니다.

두 소득 그룹의 평균 주당 근무시간에는
통계적으로 유의한 차이가 있는 것으로 나타났습니다.

다만 이 결과는 근무시간이 높은 소득의 직접적인 원인이라는 의미가 아니라,
두 변수 사이에 차이가 관찰되었다는 의미입니다.

### RandomForest 평가

- Accuracy: 0.8565
- Precision: 0.7331
- Recall: 0.6358
- F1-score: 0.6810

전체 정확도는 약 85.65%였지만,
실제 고소득자 중 일부를 저소득자로 예측하는 문제가 나타났습니다.

고소득 그룹의 Precision은 73.31%였지만 Recall은 63.58%로 더 낮았습니다.

따라서 Accuracy만 확인하지 않고,
고소득 그룹의 Precision, Recall, F1-score를 함께 확인해야 합니다.

---

## 테스트 및 코드 품질

### 테스트 실행

```bash
python -m pytest -v
```

현재 테스트에서는 다음 내용을 확인합니다.

- Pandas와 Polars가 원본 데이터를 누락이나 추가 행 없이 정상 크기인 `(32561, 15)`로 불러오는지 확인
- Pandas와 Polars가 동일한 컬럼 이름과 순서로 데이터를 해석하는지 확인
- 전처리 후 결측치가 0개인지 확인
- 전처리 후 중복 행이 0개인지 확인
- 원본 데이터에서 중복 행 24개가 제거되었는지 확인
- 전처리 결과 CSV 파일이 정상적으로 생성되는지 확인

### 코드 품질 검사

```bash
ruff check .
```

현재 테스트 결과:

```text
2 passed
```

현재 Ruff 검사 결과:

```text
All checks passed!
```

---

## 개발 중 발생한 문제와 해결

### 1. Pandas와 Polars의 데이터 크기 불일치

Polars가 원본 파일 마지막의 빈 줄을 하나의 데이터 행으로 인식하여,
처음에는 Pandas보다 한 행이 많게 나타났습니다.

모든 값이 비어 있는 행을 제거한 뒤
두 데이터의 크기가 모두 `(32561, 15)`가 되도록 수정했습니다.

### 2. 문자열 처리 과정의 경고

Pandas에서 모든 컬럼에 문자열 처리를 적용하는 과정에서 경고가 발생했습니다.

문자열 자료형 컬럼만 선택한 뒤
앞뒤 공백 제거를 적용하도록 수정했습니다.

### 3. pytest에서 `src` 모듈을 찾지 못한 문제

전역 환경의 `pytest` 명령이 실행되면서 다음 오류가 발생했습니다.

```text
ModuleNotFoundError: No module named 'src'
```

다음과 같이 현재 Python 환경을 기준으로 pytest를 실행하여 해결했습니다.

```bash
python -m pytest -v
```

### 4. 보고서 해석이 고정되는 문제

초기에는 분석 결과에 대한 결론을 문자열로 직접 작성했습니다.

이 방식은 데이터나 모델 결과가 변경되어도
같은 해석이 출력될 수 있다는 문제가 있었습니다.

평균값, p-value, 상관계수, Precision과 Recall을 실제로 비교하여
실행 결과에 따라 보고서의 해석 문장이 달라지도록 수정했습니다.

### 5. 시각화 함수가 `None`을 반환한 문제

Plotly 저장 코드를 수정하는 과정에서 `return` 문이 빠져,
`create_visualizations()` 함수가 `None`을 반환했습니다.

이로 인해 보고서 생성 과정에서
시각화 경로를 사용할 수 없는 오류가 발생했습니다.

중복된 박스플롯 생성 코드를 제거하고,
생성된 시각화 파일 경로를 딕셔너리로 반환하도록 수정했습니다.

### 6. Plotly HTML이 실행할 때마다 변경되는 문제

Plotly가 HTML 내부의 그래프 ID를 실행할 때마다 새로 생성하여,
같은 데이터로 실행해도 Git에서 수정된 파일로 표시되었습니다.

고정된 `div_id`를 지정하여
동일한 입력에서는 동일한 HTML이 생성되도록 수정했습니다.

---

## 분석 결과에 대한 의견

- 기능별로 파일을 나누니 코드가 길어져도 각 부분의 역할을 찾고 수정하기 쉬웠습니다.
- 전체 Pipeline을 한 번에 실행할 수 있어 분석 과정의 누락 여부를 확인하기 편했습니다.
- 모델의 전체 정확도는 높았지만 실제 고소득자를 놓치는 경우가 비교적 많았습니다.

## 개선방안

- Accuracy만 확인하지 않고 Precision, Recall, F1-score를 함께 확인해야 한다고 생각했습니다.
- 이후 통계 분석, 모델 학습, 시각화, 보고서 생성 함수에 대한 테스트도 추가할 필요가 있습니다.
- RandomForest 외에 Logistic Regression이나 Gradient Boosting 등을 적용하여 학습 시간과 예측 성능을 비교할 수 있습니다.
- `n_estimators`, `max_depth` 등의 하이퍼파라미터를 변경하면서 모델 성능과 파일 크기가 어떻게 달라지는지 비교할 수 있습니다.
- Pandas와 Polars의 로딩 시간과 메모리 사용량을 동일한 조건에서 측정하여 성능 차이를 추가로 비교할 수 있습니다.
- 클래스 불균형을 고려하여 `class_weight` 적용 전후의 Precision, Recall, F1-score를 비교할 수 있습니다.