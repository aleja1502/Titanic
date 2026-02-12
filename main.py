from typing import Literal
import pandas as pd

EstadoDato = Literal['valido', 'outlier_severo', 'outlier_leve', 'error']

def clasificar_observacion(obs: dict) -> EstadoDato:
    if 'valor' in obs and obs['valor'] is None:
        return 'error'
    if 'z_score' in obs and abs(obs['z_score']) > 3:
        return 'outlier_severo'
    if 'z_score' in obs and abs(obs['z_score']) > 2:
        return 'outlier_leve'
    return 'valido'


df = pd.read_csv("data/titanic.csv")

columna = "Age"

media = df[columna].mean()
desv = df[columna].std()

df["z_score"] = (df[columna] - media) / desv

def clasificar_fila(row):
    if pd.isna(row[columna]):
        obs = {'valor': None}
    else:
        obs = {'z_score': row['z_score']}
    return clasificar_observacion(obs)

df["estado_dato"] = df.apply(clasificar_fila, axis=1)

print(df[[columna, "z_score", "estado_dato"]].head())
print(df["estado_dato"].value_counts())
