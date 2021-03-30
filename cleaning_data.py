from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


def div_into_ftr_form(
    df: pd.DataFrame,
    is_test: bool = False,
    dir: Path = Path('features'),
) -> None:
    for column in df.columns:
        if not is_test:
            pd.DataFrame(df[column]).to_feather(dir/(column+'_train.ftr'))
        else:
            pd.DataFrame(df[column]).to_feather(dir/(column+'_test.ftr'))


if __name__ == '__main__':
    trn = pd.read_csv('train_ver2.csv')
    tst = pd.read_csv('test_ver2.csv')

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
    df['antiguedad'].replace('     NA', -99, inplace=True)
    df['antiguedad'] = df['antiguedad'].astype(np.int8)

    df['renta'].replace(' NA', -99, inplace=True)
    df['renta'].replace('         NA', -99, inplace=True)
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

    # dfから学習データと検証データを取り出し
    trn = df[:len(trn)]
    tst = df[len(trn):]

    # indexを列として保存
    features += ['idx']
    df_trn = trn.copy()
    df_tst = tst.copy()
    df_trn['idx'] = trn.index
    df_tst['idx'] = tst.index

    # 速度が早いfeather形式で列ごとに保存
    df_trn = df_trn.reset_index()
    df_tst = df_tst.reset_index()

    div_into_ftr_form(df_trn, is_test=False)
    div_into_ftr_form(df_tst, is_test=True)
