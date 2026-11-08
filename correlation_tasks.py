"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations
import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep='\t', na_values=['NA'])
    
    males = df[df['Gender'] == 'Male']
    females = df[df['Gender'] == 'Female']
    
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    
    men_corr = {}
    women_corr = {}
    
    max_abs_corr = -1.0
    strongest_feature = ""
    
    for feature in features:
        m_corr = males[feature].corr(males['MRI_Count'])
        f_corr = females[feature].corr(females['MRI_Count'])
        
        men_corr[feature] = float(m_corr)
        women_corr[feature] = float(f_corr)
        
        if abs(m_corr) > max_abs_corr:
            max_abs_corr = abs(m_corr)
            strongest_feature = feature
            
        if abs(f_corr) > max_abs_corr:
            max_abs_corr = abs(f_corr)
            strongest_feature = feature
            
    return BrainCorrelationSummary(
        men_count=len(males),
        women_count=len(females),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest_feature
    )
