import pandas as pd
import numpy as np
import xgboost as xgb

np.random.seed(2018)

trn = pd.read_csv('train_ver2.csv')
tst = pd.read_csv('test_ver2.csv')

# 前処理
prods = trn.columns[24:].tolist()

# 欠損値に0を代入
trn[prods] = trn[prods].fillna(0.0).astype(np.int8)

no_product = trn[prods].sum(axis=1) == 0
trn = trn[~no_product]

for col in trn.columns[24:]:
    tst[col] = 0
df = pd.concat([trn, tst], axis=0)

features = []

categorical_cols = [
    'ind_empleado',
    'pais_residencia',
    'sexo',
    'tiprel_1mes',
    'indresi',
    'indext',
    'conyuemp',
    'canal_entrada',
    'indfall',
    'tipodom',
    'nomprov',
    'segmento'
]

for col in categorical_cols:
    df[col], _ = df[col].factorize(na_sentinel=-99)
features += categorical_cols

# 前処理
df['age'].replace(' NA', -99, inplace=True)
df['age'] = df['age'].astype(np.int8)


df['antiguedad'].replace(' NA', -99, inplace=True)
df['antiguedad'] = df['antiguedad'].astype(np.int8)

df['renta'].replace(' NA', -99, inplace=True)
df['renta'].fillna(-99, inplace=True)
df['renta'] = df['renta'].astype(float).astype(np.int8)

df['indrel_1mes'].replace('P', 5, inplace=True)
df['indrel_1mes'].fillna(-99, inplace=True)
df['indrel_1mes'] = df['indrel_1mes'].astype(float).astype(np.int8)

features += [
    'age',
    'antiguedad',
    'renta',
    'ind_nuevo',
    'indrel',
    'indrel_1mes',
    'ind_actividad_cliente'
]

df[features].to_feather('train_ver2.fth')
