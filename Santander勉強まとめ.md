# Santander勉強まとめ

##　背景

サンタンデール銀行：スペインのサンタンデールグループが所有する銀行

顧客に対し，多様な金融商品を販売し，利益を得ている．

銀行に訪れる顧客にマッチした金融商品を推薦したい．

## 評価基準

顧客が新規に購買する商品はなんなのか，を予測

評価指標：MAP＠７（Mean Average Precision @ 7)

​	Average Precision：**適合率の平均**

​	例：７個の金融商品を予測する場合

​	1は正答，0は誤答を表す．

```python
# Prediction
1 0 0 1 1 1 0
```

​	予測結果について適合率を計算すると，

​	最初の１個を予測したときの適合率は1/1 = 100%

​	２回目と３回目は誤答なので0%

​	４回目に正答した場合は2/4 = 50%…

```python
# Precision（適合率）
1/1 0 0 2/4 3/5 4/6 0
```

​	Average Precisionは，適合率の合計を正答の個数で割った値

```python
# Average Precision（適合率の平均）
(1/1 + 2/4 + 3/5 + 4/6)/4 = 0.69
```

​	

​	Mean Average Precision：すべての予測結果のAverage Precisionの平均値

​	@7がついているのは，最大7個の金融商品を予測することができるという意味



MAP@7は**予測の順序に非常に敏感**

- 予測のはじめの4個が正解だった場合

```python
# Prediction
1 1 1 1 0 0 0

# Precision
1/1 2/2 3/3 4/4 0 0 0

# Average Precision
(1/1 + 2/2 + 3/3 + 4/4) / 4 = 1.00
```

- 予測のおわりの4個が正解だった場合

```python
# Prediction
0 0 0 1 1 1 1

# Precision
0 0 0 1/4 2/5 3/6 4/7

# Average Precision
(1/4 + 2/5 + 3/6 + 4/7) / 4 = 0.43
```



## EDA

- データの大きさ
  - 訓練データには総計13647309個の顧客データ
  - 顧客ごとに48個の変数
  
- 変数
  - 24個の顧客関連の変数
  - 24個の金融商品の変数
  
- 注意
  - データタイプを適切に整理して，読み込む必要あり
    - age
      - 年齢のデータなのに整数型じゃない
      - ' 12'みたいな数字の入力がある
    - ncodpers
      - 顧客固有識別番号
      - 学習させる意味なし
    - ind_nuevo
      - 新規顧客指標
      - float64→int64
    - indrel_1mes
      - 月始めを基準とした顧客等級
      - 1と1.0が異なる値として存在
      - 謎のPという値あり．数字に変換するよろし
  
  
  
- 可視化
  
  - 月別金融商品保有データの相対的累積棒グラフ<img src="/home/taru/src/kaggle/Santander_Product_Recommendation/月別金融商品保有データの相対的累積棒グラフ.png" alt="月別金融商品保有データの相対的累積棒グラフ" style="zoom:200%;" />
  
  
  
- ***テーブルデータコンペだと変数ごとに気づいたことを書き込んだ分析ノートを作るべし***
  
  - 変数名，説明，データタイプ，特徴，アイディア
  
- **探索的データ分析（EDA）で得たいもの**
  
  1. データの基礎統計と可視化によってデータを直接目で確認すること
  2. 分析のアイディアを見つけ出すこと
  3. 予測変数の特徴を見つけ出すこと
  
  
  
- インサイト
  
  - 今回のコンペで予測したい値は顧客が新規に購買する商品についてである
    - 購入後持続的に該当商品を保有しているかどうかはあんま関心を割く必要なし
  - 月別の新規購買データの可視化![月別新規購買データの相対的累積棒フラグ](/home/taru/src/kaggle/Santander_Product_Recommendation/月別新規購買データの相対的累積棒フラグ.png)
    - 当座預金(ind_cco_fin_ult1)は，夏に最も高くなり，冬には縮小する
    - 短期預金(ind_deco_fin_ult1)は，6-28で特に高くなり他の時期は非常に低い
    - 給与，年金(ind_nomina_ult1, ind_nom_pens_ult1)は，夏に最も低くなり，冬に最も高くなる
    - 新規購買頻度の上位5つの金融商品は，
      - 当座預金（ind_cco_fin_ult1）
      - 信用カード（ind_tjcr_fin_ult1）
      - 給与（ind_nomina_ult1）
      - 年金（ind_nom_pens_ult1）
      - デビッドカード（ind_recibo_fin_ult1）
  
  
  
- 結論

  - 新規購入される金曜商品には季節によって変化する
    - **訓練データを何月に指定するかにより，機械学習モデルの結果が大きく異なることが予想される**
  - 取る戦略として，以下の２つが考えられる
    - 季節の変動性をモデリングする1つの一般的なモデルを構築する
    - 季節によって異なる多数のモデルを構築してそれを混合する　←◎精度よし
  - *実務で考慮するべきこと*（↓のトレードオフを理解して決定すべし）
    - 多数のモデルを季節ごとに構築して得られる性能の改善幅
    - 多数のモデルをリアルタイムで運営する費用とリスク

  

