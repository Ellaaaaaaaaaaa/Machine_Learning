# 数据分析
import pandas as pd
# 处理多维数组和矩阵运算
import numpy as np
import scipy
from autogluon.tabular import TabularPredictor
import pickle

data = pd.read_feather('house_sales.ftr')
scipy.__version__, np.__version__
df = data[['Sold Price', 'Sold On', 'Type', 'Year built', 'Bedrooms', 'Bathrooms']].copy()
c = 'Sold Price'
# 对数据进行特征处理
if c in df.select_dtypes('object').columns:
    # 提取索引为c的列，将列里单位去掉，做对数变换
    df.loc[:, c] = np.log10(
        pd.to_numeric(df[c].replace(r'[$,-]', '', regex=True)) + 1)
df = df[(df['Sold Price'] >= 4) & (df['Sold Price'] <= 8)]
test_start, test_end = pd.Timestamp(2021, 2, 15), pd.Timestamp(2021, 3, 1)
train_start = pd.Timestamp(2021, 1, 1)
df['Sold On'] = pd.to_datetime(df['Sold On'], errors='coerce')
train = df[(df['Sold On'] >= train_start) & (df['Sold On'] < test_start)]
test = df[(df['Sold On'] >= test_start) & (df['Sold On'] < test_end)]
train.shape, test.shape


def rmsle(y_hat, y):
    # we already used log prices before, so we only need to compute RMSE 均方根误差
    return sum((y_hat - y) ** 2 / len(y)) ** 0.5


label = 'Sold Price'
predictor = TabularPredictor(label=label).fit(train)
predictor.leaderboard(test, silent=True)  # 生成排行榜
predictor.feature_importance(test)  # 计算一种特性的重要性
preds = predictor.predict(test.drop(columns=[label]))  # 预测结果
num = rmsle(preds, test[label])
print(num)

# with open('AutogluonModels/ag-20230512_191414/utils/data/X_val.pkl','rb') as f:
#     data=pickle.load(f)
# print(data)
