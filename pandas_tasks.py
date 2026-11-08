"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    df = pd.read_csv(data.csv_path)
    
    missing = df.isnull().sum().to_dict()
    
    older = int((df['Age'] > 30).sum())
    
    mean_age = df.groupby('Pclass')['Age'].mean().to_dict()
    mean_age = {int(k): v for k, v in mean_age.items()}
    
    survival = df.groupby('Pclass')['Survived'].mean().to_dict()
    survival = {int(k): v for k, v in survival.items()}
    
    top_fares = df['Fare'].nlargest(5).tolist()
    
    return TitanicSummary(
        row_count=len(df),
        missing_by_column=missing,
        adults_over_30_count=older,
        mean_age_by_pclass=mean_age,
        survival_rate_by_pclass=survival,
        highest_fares=top_fares
    )
