"""전처리와 RandomForest 모델을 하나의 Pipeline으로 구성하는 모듈."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_loader import load_config


def train_model(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
    model_path: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
) -> tuple[Pipeline, dict]:
    """전처리 Pipeline과 RandomForest를 학습하고 평가한다."""

    feature_columns = numeric_columns + categorical_columns
    X = df[feature_columns]
    y = df[target_column]

    # income 비율이 학습·테스트 데이터에 비슷하게 유지되도록 stratify를 적용한다.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 학습 데이터에서 가장 많은 클래스로만 예측하는 기준 모델이다.
    # 복잡한 모델이 단순 예측보다 실제로 나은지 비교하기 위해 사용한다.
    baseline_model = DummyClassifier(
        strategy="most_frequent",
    )

    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)

    baseline_class = str(
        y_train.value_counts().idxmax()
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions,
    )
    baseline_precision = precision_score(
        y_test,
        baseline_predictions,
        pos_label=">50K",
        zero_division=0,
    )
    baseline_recall = recall_score(
        y_test,
        baseline_predictions,
        pos_label=">50K",
        zero_division=0,
    )
    baseline_f1 = f1_score(
        y_test,
        baseline_predictions,
        pos_label=">50K",
        zero_division=0,
    )

    # 수치형 결측치는 중앙값으로 대체한다.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # 범주형 결측치를 최빈값으로 대체한 뒤 원-핫 인코딩한다.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    # 전처리와 모델 학습을 하나의 Pipeline으로 연결한다.
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model_pipeline.fit(X_train, y_train)
    predictions = model_pipeline.predict(X_test)

    model_accuracy = accuracy_score(
        y_test,
        predictions,
    )
    model_precision = precision_score(
        y_test,
        predictions,
        pos_label=">50K",
        zero_division=0,
    )
    model_recall = recall_score(
        y_test,
        predictions,
        pos_label=">50K",
        zero_division=0,
    )
    model_f1 = f1_score(
        y_test,
        predictions,
        pos_label=">50K",
        zero_division=0,
    )

    metrics = {
        "accuracy": model_accuracy,
        "precision": model_precision,
        "recall": model_recall,
        "f1": model_f1,
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
        ),
        "baseline": {
            "strategy": "most_frequent",
            "predicted_class": baseline_class,
            "accuracy": baseline_accuracy,
            "precision": baseline_precision,
            "recall": baseline_recall,
            "f1": baseline_f1,
        },
        "accuracy_improvement": (
            model_accuracy - baseline_accuracy
        ),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    # 전처리를 포함한 전체 Pipeline 객체를 joblib 파일로 저장한다.
    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, output_path)

    metrics["model_path"] = str(output_path)

    return model_pipeline, metrics


if __name__ == "__main__":
    config = load_config()
    df = pd.read_csv(config["data"]["processed_path"])

    _, results = train_model(
        df=df,
        numeric_columns=config["columns"]["numeric"],
        categorical_columns=config["columns"]["categorical"],
        target_column=config["columns"]["target"],
        model_path=config["paths"]["model"],
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        n_estimators=config["model"]["n_estimators"],
    )

    print(f"학습 데이터 수: {results['train_size']}")
    print(f"테스트 데이터 수: {results['test_size']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-score: {results['f1']:.4f}")

    print("\n[Confusion Matrix]")
    print(results["confusion_matrix"])

    print("\n[Classification Report]")
    print(results["classification_report"])

    print(f"모델 저장 위치: {results['model_path']}")

    baseline = results["baseline"]

    print("\n[다수 클래스 베이스라인]")
    print(
        "예측 클래스: "
        f"{baseline['predicted_class']}"
    )
    print(
        "Baseline Accuracy: "
        f"{baseline['accuracy']:.4f}"
    )
    print(
        "Baseline Precision: "
        f"{baseline['precision']:.4f}"
    )
    print(
        "Baseline Recall: "
        f"{baseline['recall']:.4f}"
    )
    print(
        "Baseline F1-score: "
        f"{baseline['f1']:.4f}"
    )

    print("\n[RandomForest와 베이스라인 비교]")
    print(
        "Accuracy 개선폭: "
        f"{results['accuracy_improvement']:.4f} "
        f"({results['accuracy_improvement'] * 100:.2f}%p)"
    )