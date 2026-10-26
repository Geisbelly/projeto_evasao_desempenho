"""Experimentos reproduzíveis para o projeto de evasão e desempenho acadêmico.

A base é obtida pelo pacote ucimlrepo. O conjunto de teste fica reservado para
a avaliação final; busca de hiperparâmetros e CV usam somente o treino.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_CANDIDATES = ["Target", "target", "class", "Class", "Situation", "situacao"]


def load_data():
    dataset = fetch_ucirepo(id=697)
    X = dataset.data.features.copy()
    y_df = dataset.data.targets.copy()
    if y_df.shape[1] != 1:
        raise ValueError(f"Esperado um alvo; encontrados: {list(y_df.columns)}")
    y = y_df.iloc[:, 0].astype(str).str.strip()
    X.columns = [str(c).strip() for c in X.columns]
    # Normaliza rótulos, preservando os nomes informados pela fonte.
    return X, y, dataset


def feature_sets(X):
    """Seleciona atributos por famílias sem usar variáveis do segundo semestre."""
    first_semester_prefixes = (
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
    )
    second_semester_prefixes = (
        "Curricular units 2nd sem",
    )
    first_sem_cols = [c for c in X.columns if c.startswith(first_semester_prefixes)]
    second_sem_cols = [c for c in X.columns if c.startswith(second_semester_prefixes)]

    # Atributos de ingresso / contexto documentados no conjunto original.
    # O nome exato das colunas é verificado contra os dados carregados.
    entry_candidates = [
        "Marital status", "Application mode", "Application order",
        "Course", "Daytime/evening attendance", "Previous qualification",
        "Previous qualification (grade)", "Nacionality", "Nationality",
        "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation", "Admission grade",
        "Displaced", "Educational special needs", "Debtor",
        "Tuition fees up to date", "Gender", "Scholarship holder",
        "Age at enrollment", "International",
        "Unemployment rate", "Inflation rate", "GDP",
    ]
    entry_cols = [c for c in entry_candidates if c in X.columns]
    # As colunas de 2º semestre são deliberadamente excluídas de ambos cenários.
    if not entry_cols:
        raise ValueError("Não foi possível identificar atributos de ingresso. Verifique os nomes das colunas.")
    scenario_a = X[entry_cols].copy()
    scenario_b_cols = list(dict.fromkeys(entry_cols + first_sem_cols))
    scenario_b = X[scenario_b_cols].copy()
    if not first_sem_cols:
        warnings.warn("Nenhuma coluna de 1º semestre foi encontrada; verifique os nomes da base.")
    return scenario_a, scenario_b, entry_cols, first_sem_cols, second_sem_cols


def make_pipeline(X, model):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric))
    if categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, categorical))
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("prep", prep), ("model", model)])


def model_specs():
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "DecisionTree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            None,
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": [200, 400], "model__max_depth": [None, 10, 20],
             "model__min_samples_leaf": [1, 3]},
        ),
        "SVM": (
            SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE),
            None,
        ),
    }


def run_scenario(name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    details = {}
    for model_name, (model, param_grid) in model_specs().items():
        pipe = make_pipeline(X_train, model)
        if param_grid:
            search = GridSearchCV(
                pipe, param_grid=param_grid, scoring="f1_macro", cv=cv,
                n_jobs=-1, refit=True, return_train_score=False
            )
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            best_params = search.best_params_
            cv_score = float(search.best_score_)
            cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
        else:
            cv_results = cross_validate(
                pipe, X_train, y_train, cv=cv, scoring="f1_macro",
                n_jobs=-1, return_train_score=False
            )
            cv_score = float(np.mean(cv_results["test_score"]))
            cv_std = float(np.std(cv_results["test_score"], ddof=1))
            fitted = pipe.fit(X_train, y_train)
            best_params = "Não foi realizada busca neste modelo"
        pred = fitted.predict(X_test)
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, pred, labels=sorted(y.unique()))
        rows.append({
            "scenario": name, "model": model_name,
            "cv_macro_f1_mean": cv_score, "cv_macro_f1_std": cv_std,
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
            "test_dropout_recall": float(report.get("Dropout", {}).get("recall", np.nan)),
            "best_params": json.dumps(best_params, ensure_ascii=False) if isinstance(best_params, dict) else best_params,
        })
        details[model_name] = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "labels": sorted(y.unique()),
            "best_params": best_params,
        }
    return rows, details


def main():
    X, y, dataset = load_data()
    print("Dimensão X:", X.shape)
    print("Distribuição do alvo:\n", y.value_counts(dropna=False))
    print("Ausentes por coluna:\n", X.isna().sum().sort_values(ascending=False).head(15))
    print("Duplicatas exatas:", int(X.join(y.rename("TARGET")).duplicated().sum()))

    scenario_a, scenario_b, entry_cols, first_cols, second_cols = feature_sets(X)
    print("\nAtributos cenário A (ingresso):", entry_cols)
    print("Atributos do 1º semestre:", first_cols)
    print("Atributos do 2º semestre excluídos:", second_cols)

    all_rows = []
    all_details = {}
    for scenario_name, features in [
        ("A_ingresso", scenario_a),
        ("B_ingresso_mais_1o_semestre", scenario_b),
    ]:
        rows, details = run_scenario(scenario_name, features, y)
        all_rows.extend(rows)
        all_details[scenario_name] = details

    results = pd.DataFrame(all_rows)
    results.to_csv(OUT / "model_comparison.csv", index=False)
    with open(OUT / "classification_details.json", "w", encoding="utf-8") as f:
        json.dump(all_details, f, ensure_ascii=False, indent=2, default=str)
    metadata = {
        "dataset_id": 697,
        "n_rows": int(X.shape[0]),
        "n_predictors": int(X.shape[1]),
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
        "scenario_A_features": entry_cols,
        "first_semester_features": first_cols,
        "excluded_second_semester_features": second_cols,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "primary_metric": "f1_macro",
    }
    with open(OUT / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("\nResultados:")
    print(results.sort_values(["scenario", "cv_macro_f1_mean"], ascending=[True, False]).to_string(index=False))
    print(f"\nArquivos salvos em: {OUT}")


if __name__ == "__main__":
    main()
