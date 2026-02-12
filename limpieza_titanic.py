import pandas as pd
import numpy as nppython
from typing import List, Optional

def eliminar_duplicados(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    n_antes = len(df)
    df_limpio = df.drop_duplicates(subset=subset, keep='first')
    n_despues = len(df_limpio)
    print(f"✅ Eliminados {n_antes - n_despues} duplicados")
    return df_limpio

def imputar_nulos(df: pd.DataFrame, columnas: List[str], estrategia: str = 'mediana') -> pd.DataFrame:
    df_limpio = df.copy()
    for col in columnas:
        if estrategia == 'media':
            valor = df_limpio[col].mean()
        elif estrategia == 'mediana':
            valor = df_limpio[col].median()
        else:
            valor = df_limpio[col].mode()[0]
        
        n_nulos = df_limpio[col].isna().sum()
        df_limpio[col] = df_limpio[col].fillna(valor)
        print(f"✅ {col}: {n_nulos} nulos imputados con {estrategia}={valor:.2f}")
    
    return df_limpio


def detectar_outliers_iqr(df: pd.DataFrame, columna: str, factor: float = 1.5) -> pd.Series:
    """
    Detecta outliers usando el método IQR.
    
    Returns:
        Serie booleana (True = outlier)
    """
    Q1, Q3 = df[columna].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    limite_inf = Q1 - factor * IQR
    limite_sup = Q3 + factor * IQR
    
    outliers = (df[columna] < limite_inf) | (df[columna] > limite_sup)
    print(f"⚠️ {columna}: {outliers.sum()} outliers detectados (IQR x{factor})")
    return outliers

