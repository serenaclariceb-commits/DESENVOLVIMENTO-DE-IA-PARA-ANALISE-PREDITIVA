from __future__ import annotations

from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def find_dataset_path() -> Path:
    candidates = [
        DATA_DIR / "manutencao_preditiva.csv",
        DATA_DIR / '"manutencao_preditiva.csv"',
        DATA_DIR / "'manutencao_preditiva.csv'",
        BASE_DIR / "manutencao_preditiva.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Arquivo de dados não encontrado. Coloque o CSV em 'preditor-falhas-industria/data/'"
    )


def load_dataset() -> pd.DataFrame:
    path = find_dataset_path()
    df = pd.read_csv(path)
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()

    if "velocidade_rotacao_rpm" in df.columns and "torque_nm" in df.columns:
        df["potencia"] = (
            df["velocidade_rotacao_rpm"]
            * df["torque_nm"]
            * (2 * 3.141592653589793 / 60)
            / 1000
        )

    if "tipo" in df.columns:
        df = pd.get_dummies(df, columns=["tipo"], dtype=int)

    return df


def build_model_data(df: pd.DataFrame):
    target_col = "falha_maquina"
    excluded_cols = {
        target_col,
        "udi",
        "id_produto",
        "falha_twf",
        "falha_hdf",
        "falha_pwf",
        "falha_osf",
        "falha_rnf",
    }
    feature_cols = [
        col for col in df.columns if col not in excluded_cols
    ]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    imputation_values = X_train.median()
    columns_without_median = imputation_values[imputation_values.isna()].index.tolist()
    if columns_without_median:
        raise ValueError(
            "Não foi possível calcular a mediana para estas colunas do treino: "
            + ", ".join(columns_without_median)
        )

    X_train = X_train.fillna(imputation_values)
    X_test = X_test.fillna(imputation_values)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    return X_train_bal, y_train_bal, X_test, y_test, X_train, y_train


def evaluate_models(df: pd.DataFrame | None = None):
    if df is None:
        df = preprocess_dataset(load_dataset())

    X_train_bal, y_train_bal, X_test, y_test, X_train, y_train = (
        build_model_data(df)
    )

    models = {}
    results = []
    candidates = [
        (
            "KNN",
            f"K = {k}",
            make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=k),
            ),
        )
        for k in [3, 5, 7]
    ]
    candidates.extend(
        (
            "Árvore de Decisão",
            f"max_depth = {depth}",
            DecisionTreeClassifier(max_depth=depth, random_state=42),
        )
        for depth in [3, 5, None]
    )

    for model_name, parameter, model in candidates:
        model.fit(X_train_bal, y_train_bal)
        train_predictions = model.predict(X_train_bal)
        test_predictions = model.predict(X_test)
        key = f"{model_name} ({parameter})"
        models[key] = model
        results.append(
            {
                "modelo": model_name,
                "parametro": parameter,
                "acuracia_treino": accuracy_score(
                    y_train_bal, train_predictions
                ),
                "acuracia_teste": accuracy_score(y_test, test_predictions),
                "precision_falha": precision_score(
                    y_test, test_predictions, zero_division=0
                ),
                "recall_falha": recall_score(
                    y_test, test_predictions, zero_division=0
                ),
                "f1_falha": f1_score(
                    y_test, test_predictions, zero_division=0
                ),
            }
        )

    baseline_label = int(y_train.mode().iloc[0])
    baseline_accuracy = accuracy_score(
        y_test, [baseline_label] * len(y_test)
    )
    return (
        pd.DataFrame(results),
        models,
        X_test,
        y_test,
        baseline_label,
        baseline_accuracy,
    )


def train_model() -> tuple:
    (
        results,
        models,
        X_test,
        y_test,
        baseline_label,
        baseline_accuracy,
    ) = evaluate_models()
    best_result = results.loc[results["acuracia_teste"].idxmax()]
    best_key = (
        f"{best_result['modelo']} ({best_result['parametro']})"
    )
    model = models[best_key]
    y_pred = model.predict(X_test)

    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        f"\nBaseline (sempre classe {baseline_label}): "
        f"{baseline_accuracy:.4f}"
    )
    if best_result["acuracia_teste"] > baseline_accuracy:
        print(f"Melhor candidato: {best_key} (supera o baseline de acurácia).")
    else:
        print(
            f"Melhor candidato: {best_key}, mas não supera o baseline; "
            "nenhum modelo foi aprovado."
        )
    print("\nRelatório do melhor candidato:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))
    return model, X_test, y_test


if __name__ == "__main__":
    train_model()
