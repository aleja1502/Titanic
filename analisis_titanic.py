from limpieza_titanic import eliminar_duplicados, imputar_nulos, detectar_outliers_iqr
import pandas as pd

df = pd.read_csv("data/titanic.csv")

df = eliminar_duplicados(df)

df = imputar_nulos(
    df,
    ['Age', 'Fare', 'SibSp', 'Parch'],
    estrategia='mediana'
)

outliers = detectar_outliers_iqr(df, 'Age')
