from pathlib import Path

import pandas as pd


def load_datasets(
    feats: list,
    dir: Path = Path('features')
) -> {pd.DataFrame, pd.DataFrame}:
    list_df = [pd.read_feather(dir / f'{f}_train.ftr') for f in feats]
    train = pd.concat(list_df, axis=1)
    train.set_index('idx', inplace=True)
    train.index.name = None

    list_df = [pd.read_feather(dir / f'{f}_test.ftr') for f in feats]
    test = pd.concat(list_df, axis=1)
    test.set_index('idx', inplace=True)
    test.index.name = None
    return train, test


if __name__ == '__main__':
    features = [
        'ind_empleado', 'pais_residencia',
        'sexo', 'tiprel_1mes', 'indresi',
        'indext', 'conyuemp', 'canal_entrada',
        'indfall', 'tipodom', 'nomprov',
        'segmento', 'age', 'antiguedad',
        'renta', 'ind_nuevo', 'indrel',
        'indrel_1mes', 'ind_actividad_cliente', 'idx'
    ]
    trn, tst = load_datasets(features)

    print(trn)
    print(tst)
