import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def check_columns(df: pd.DataFrame) -> None:
    print(f'Shape: {df.shape}')
    print(f'Head: {df.head()}')
    for col in df.columns:
        print(f'{df[col].head()}\n')

    print(f'Info: {df.info()}')

    # 数値変数をチェック
    num_cols = [col for col in df.columns[:24]
                if df[col].dtype in ['int64', 'float64']]
    print(f'num_cols Desc: {df[num_cols].describe()}')

    # カテゴリ変数をチェック
    cat_cols = [col for col in df.columns[:24] if df[col].dtype in ['O']]
    print(f'cat_cols Desc: {df[cat_cols].describe()}')

    # カテゴリ変数のユニークな値をチェック
    for col in cat_cols:
        uniq = np.unique(df[col].astype(str))
        print('-'*100)
        print(f'# col {col}, n_uniq {len(uniq)}, uniq {uniq}')


# 変数ごとの値を棒グラフで可視化（データクレンジングなし）
def plot_hist(df: pd.DataFrame):
    skip_cols = ['ncodpers', 'renta']
    for col in df.columns:
        if col in skip_cols:
            continue

        print('-'*100)
        print(f'col: {col}')

        f, ax = plt.subplots(figsize=(20, 15))
        sns.countplot(x=col, data=df, alpha=0.5)
        plt.show()


# 月別の金融商品の保有データを月別相対値で可視化
def plot_cumulative_bar_graph(df: pd.DataFrame):
    months = df['fecha_dato'].unique().tolist()
    label_cols = df.columns[24:].tolist()

    label_over_time = []
    for i in range(len(label_cols)):
        # 月ごとに各商品の合計を計算
        label_sum = df.groupby(['fecha_dato'])[label_cols[i]].agg('sum')
        label_over_time.append(label_sum.tolist())

    label_sum_over_time = []
    for i in range(len(label_cols)):
        # n番目の商品の出力を1~n番目の商品の合計として集計しなおし
        label_sum_over_time.append(np.array(label_over_time[i:]).sum(axis=0))

    color_list = ['#F5B7B1', '#D2B4DE', '#AED6F1', '#A2D9CE',
                  '#ABEBC6', '#F9E79F', '#F5CBA7', '#CCD1D1']

    label_sum_percent = (
        label_sum_over_time / (1.*np.asarray(label_sum_over_time).max(axis=0))
    )*100

    f, ax = plt.subplots(figsize=(30, 15))
    for i in range(len(label_cols)):
        sns.barplot(
            x=months,
            y=label_sum_percent[i],
            color=color_list[i % 8],
            alpha=0.7
        )

    plt.legend(
        [plt.Rectangle((0, 0), 1, 1, fc=color_list[i % 8], edgecolor='none')
         for i in range(len(label_cols))],
        label_cols,
        loc=1,
        ncol=1,
        prop={'size': 14},
        bbox_to_anchor=(1.15, 1)
    )

    plt.show()


# 24個の金融商品の新規購買データを作成
def create_labels():
    df = pd.read_csv('train_ver2.csv')
    prods = df.columns[24:].tolist()

    def date_to_int(str_date):
        Y, M, D = [int(a)for a in str_date.strip().split('-')]
        int_date = (int(Y)-2015)*12 + int(M)
        return int_date

    df['int_date'] = df['fecha_dato'].map(date_to_int).astype(np.int8)

    df_lag = df.copy()
    df_lag['int_date'] += 1
    df_lag.columns = [
        col+'_prev' if col not in ['ncodpers', 'int_date']
        else col for col in df.columns
    ]

    print('merging!')
    df_trn = df.merge(df_lag, on=['ncodpers', 'int_date'], how='left')
    print('merged!')

    del df, df_lag

    for prod in prods:
        prev = prod+'_prev'
        df_trn[prev].fillna(0, inplace=True)

    for prod in prods:
        padd = prod + '_add'
        prev = prod + '_prev'
        df_trn[padd] = (
            (df_trn[prod] == 1) & (df_trn[prev] == 0)
        ).astype(np.int8)

    add_cols = [prod + '_add' for prod in prods]
    labels = df_trn[add_cols]
    labels.columns = prods
    labels.to_csv('labels.csv', index=False)


def visualize_labels():
    labels = pd.read_csv('labels.csv').astype(int)
    fecha_dato = pd.read_csv('train_ver2.csv', usecols=['fecha_dato'])

    labels['date'] = fecha_dato.fecha_dato
    months = np.unique(fecha_dato.fecha_dato).tolist()
    label_cols = labels.columns.tolist()[:24]

    label_over_time = []
    for i in range(len(label_cols)):
        label_over_time.append(
            labels.groupby(['date'])[label_cols[i]].agg('sum').tolist()
        )

    label_sum_over_time = []
    for i in range(len(label_cols)):
        label_sum_over_time.append(np.asarray(label_over_time[i:]).sum(axis=0))

    color_list = ['#F5B7B1', '#D2B4DE', '#AED6F1', '#A2D9CE',
                  '#ABEBC6', '#F9E79F', '#F5CBA7', '#CCD1D1']

    label_sum_percent = (
        label_sum_over_time / (1.*np.asarray(label_sum_over_time).max(axis=0))
    )*100

    f, ax = plt.subplots(figsize=(30, 15))
    for i in range(len(label_cols)):
        sns.barplot(
            x=months,
            y=label_sum_percent[i],
            color=color_list[i % 8],
            alpha=0.7
        )

    plt.legend(
        [plt.Rectangle((0, 0), 1, 1, fc=color_list[i % 8], edgecolor='none')
         for i in range(len(label_cols))],
        label_cols,
        loc=1,
        ncol=1,
        prop={'size': 14},
        bbox_to_anchor=(1.15, 1)
    )

    plt.show()


if __name__ == '__main__':
    visualize_labels()
