import pandas as pd #数据处理包
import time
import numpy as np
from datetime import datetime
from pandas import DataFrame
from sklearn.cross_validation import  train_test_split  #数据分割
from sklearn.feature_extraction import DictVectorizer   #特征转化器
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier #决策树分类器
from sklearn.ensemble import RandomForestClassifier #随机森林分类器
from sklearn.ensemble import GradientBoostingClassifier #梯度提升树分类器
from xgboost import XGBClassifier   #XGBoost分类器
from sklearn.metrics import log_loss #logloss损失函数
from sklearn.cross_validation import cross_val_score    #交叉验证
from sklearn.preprocessing import StandardScaler    #标准化数据
from sklearn.preprocessing import MinMaxScaler  #归一化数据

def item_cvr(data,train_data): #0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1,df2],axis = 0,ignore_index = True)
    df = df[['item_id','is_trade']]
    df = pd.get_dummies(df,columns=['is_trade'],prefix='label')
    df = df.groupby('item_id',as_index=False).sum()
    df['item_id' + '_sum'] = (df['label_0'] + df['label_1'])
    df['item_id' + '_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
        #print(df)
    train_data = pd.merge(train_data, df[['item_id', 'item_id' + '_sum', 'item_id' + '_cvr']], on='item_id',how='left')
    return train_data


def item_hour_cvr(data,train_data): #0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1,df2],axis = 0,ignore_index = True)
    df = df[['item_id','hour', 'is_trade']]
    df = pd.get_dummies(df,columns=['is_trade'],prefix='label')
    df = df.groupby(['item_id','hour'],as_index=False).sum()
    df['item_id' + '_hour_sum'] = (df['label_0'] + df['label_1'])   #历史点击量
    df['item_id' + '_hour_cvr'] = (df['label_1'] ) / (df['label_0'] + df['label_1'])
    train_data = pd.merge(train_data, df[['item_id','hour', 'item_id' + '_hour_sum', 'item_id' + '_hour_cvr']], on=['item_id','hour'],\
                        how='left')
    return train_data

def item_ys_cvr(data,train_data):#0.0796573567414
    df = data[data['day'] == 6]
    df = df[['item_id', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby('item_id', as_index=False).sum()
    df['item_id' + '_sum_ys'] = (df['label_0'] + df['label_1'])
    df['item_id' + '_cvr_ys'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['item_id', 'item_id' + '_sum_ys', 'item_id' + '_cvr_ys']],on='item_id',how='left')
    return train_data


def user_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    # print(df)
    df = df[['user_id', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby('user_id', as_index=False).sum()
    df['user_id' + '_sum'] = (df['label_0'] + df['label_1'])
    df['user_id' + '_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['user_id', 'user_id' + '_sum', 'user_id' + '_cvr']], on='user_id', how='left')
    return train_data


def user_hour_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['user_id', 'hour','is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id', 'hour'], as_index=False).sum()
    df['user_id' + '_hour_sum'] = (df['label_0'] + df['label_1'])  # 历史点击量
    df['user_id' + '_hour_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    train_data = pd.merge(train_data,
                        df[['user_id', 'hour', 'user_id' + '_hour_sum', 'user_id' + '_hour_cvr']],
                        on=['user_id', 'hour'], \
                        how='left')
    return train_data

def user_ys_cvr(data,train_data):  # 0.0796573567414
    df = data[data['day'] == 6]
    df = df[['user_id', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby('user_id', as_index=False).sum()
    df['user_id' + '_sum_ys'] = (df['label_0'] + df['label_1'])
    df['user_id' + '_cvr_ys'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['user_id', 'user_id' + '_sum_ys', 'user_id' + '_cvr_ys']],
                        on='user_id', how='left')
    return train_data

def shop_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['shop_id', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby('shop_id', as_index=False).sum()
    df['shop_id' + '_sum'] = (df['label_0'] + df['label_1'])
    df['shop_id' + '_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['shop_id', 'shop_id' + '_sum', 'shop_id' + '_cvr']], on='shop_id', how='left')
    return train_data


def shop_hour_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['shop_id', 'hour','is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['shop_id', 'hour'], as_index=False).sum()
    df['shop_id' + '_hour_sum'] = (df['label_0'] + df['label_1'])  # 历史点击量
    df['shop_id' + '_hour_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    train_data = pd.merge(train_data,
                        df[['shop_id', 'hour', 'shop_id' + '_hour_sum', 'shop_id' + '_hour_cvr']],
                        on=['shop_id', 'hour'], \
                        how='left')
    return train_data

def shop_ys_cvr(data,train_data):  # 0.0796573567414
    df = data[data['day'] == 6]
    df = df[['shop_id', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby('shop_id', as_index=False).sum()
    df['shop_id' + '_sum_ys'] = (df['label_0'] + df['label_1'])
    df['shop_id' + '_cvr_ys'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['shop_id', 'shop_id' + '_sum_ys', 'shop_id' + '_cvr_ys']],
                        on='shop_id', how='left')
    return train_data

def user_item_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['user_id','item_id', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','item_id'], as_index=False).sum()
    df['user_item' + '_sum'] = (df['label_0'] + df['label_1'])
    df['user_item' + '_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['user_id', 'item_id','user_item' + '_sum', 'user_item' + '_cvr']], on=['user_id','item_id'], how='left')
    return train_data

def user_item_hour_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['user_id','item_id', 'hour','is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','item_id', 'hour'], as_index=False).sum()
    df['user_item' + '_hour_sum'] = (df['label_0'] + df['label_1'])  # 历史点击量
    df['user_item' + '_hour_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    train_data = pd.merge(train_data,
                        df[['user_id','item_id', 'hour', 'user_item' + '_hour_sum', 'user_item' + '_hour_cvr']],
                        on=['user_id','item_id', 'hour'], \
                        how='left')
    return train_data

def user_item_ys_cvr(data,train_data):  # 0.0796573567414
    df = data[data['day'] == 6]
    df = df[['user_id','item_id', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','item_id'], as_index=False).sum()
    df['user_item' + '_sum_ys'] = (df['label_0'] + df['label_1'])
    df['user_item' + '_cvr_ys'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['user_id','item_id', 'user_item' + '_sum_ys', 'user_item' + '_cvr_ys']],
                        on=['user_id','item_id'], how='left')
    return train_data

def user_shop_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['user_id','shop_id', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','shop_id'], as_index=False).sum()
    df['user_shop' + '_sum'] = (df['label_0'] + df['label_1'])
    df['user_shop' + '_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['user_id', 'shop_id','user_shop' + '_sum', 'user_shop' + '_cvr']], on=['user_id','shop_id'], how='left')
    return train_data

def user_shop_hour_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['user_id','shop_id', 'hour','is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','shop_id', 'hour'], as_index=False).sum()
    df['user_shop' + '_hour_sum'] = (df['label_0'] + df['label_1'])  # 历史点击量
    df['user_shop' + '_hour_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    train_data = pd.merge(train_data,
                        df[['user_id','shop_id', 'hour', 'user_shop' + '_hour_sum', 'user_shop' + '_hour_cvr']],
                        on=['user_id','shop_id', 'hour'], \
                        how='left')
    return train_data

def user_shop_ys_cvr(data,train_data):  # 0.0796573567414
    df = data[data['day'] == 6]
    df = df[['user_id','shop_id', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','shop_id'], as_index=False).sum()
    df['user_shop' + '_sum_ys'] = (df['label_0'] + df['label_1'])
    df['user_shop' + '_cvr_ys'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['user_id','shop_id', 'user_shop' + '_sum_ys', 'user_shop' + '_cvr_ys']],
                        on=['user_id','shop_id'], how='left')
    return train_data

def user_item_price_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['user_id','item_price_level', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','item_price_level'], as_index=False).sum()
    df['user_item_price' + '_sum'] = (df['label_0'] + df['label_1'])
    df['user_item_price' + '_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['user_id', 'item_price_level','user_item_price' + '_sum', 'user_item_price' + '_cvr']], on=['user_id','item_price_level'], how='left')
    return train_data

def user_item_price_hour_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['user_id','item_price_level', 'hour','is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','item_price_level', 'hour'], as_index=False).sum()
    df['user_item_price' + '_hour_sum'] = (df['label_0'] + df['label_1'])  # 历史点击量
    df['user_item_price' + '_hour_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    train_data = pd.merge(train_data,
                        df[['user_id','item_price_level', 'hour', 'user_item_price' + '_hour_sum', 'user_item_price' + '_hour_cvr']],
                        on=['user_id','item_price_level', 'hour'], \
                        how='left')
    return train_data

def user_item_price_ys_cvr(data,train_data):  # 0.0796573567414
    df = data[data['day'] == 6]
    df = df[['user_id','item_price_level', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','item_price_level'], as_index=False).sum()
    df['user_item_price' + '_sum_ys'] = (df['label_0'] + df['label_1'])
    df['user_item_price' + '_cvr_ys'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['user_id','item_price_level', 'user_item_price' + '_sum_ys', 'user_item_price' + '_cvr_ys']],
                        on=['user_id','item_price_level'], how='left')
    return train_data

def user_item_brand_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['user_id','item_brand_id', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','item_brand_id'], as_index=False).sum()
    df['user_item_brand' + '_sum'] = (df['label_0'] + df['label_1'])
    df['user_item_brand' + '_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['user_id', 'item_brand_id','user_item_brand' + '_sum', 'user_item_brand' + '_cvr']], on=['user_id','item_brand_id'], how='left')
    return train_data

def user_item_brand_hour_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['user_id','item_brand_id', 'hour','is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','item_brand_id', 'hour'], as_index=False).sum()
    df['user_item_brand' + '_hour_sum'] = (df['label_0'] + df['label_1'])  # 历史点击量
    df['user_item_brand' + '_hour_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    train_data = pd.merge(train_data,
                        df[['user_id','item_brand_id', 'hour', 'user_item_brand' + '_hour_sum', 'user_item_brand' + '_hour_cvr']],
                        on=['user_id','item_brand_id', 'hour'], \
                        how='left')
    return train_data

def user_item_brand_ys_cvr(data,train_data):  # 0.0796573567414
    df = data[data['day'] == 6]
    df = df[['user_id','item_brand_id', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','item_brand_id'], as_index=False).sum()
    df['user_item_brand' + '_sum_ys'] = (df['label_0'] + df['label_1'])
    df['user_item_brand' + '_cvr_ys'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['user_id','item_brand_id', 'user_item_brand' + '_sum_ys', 'user_item_brand' + '_cvr_ys']],
                        on=['user_id','item_brand_id'], how='left')
    return train_data

def user_item_category_0_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['user_id','category_0', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','category_0'], as_index=False).sum()
    df['user_category' + '_sum'] = (df['label_0'] + df['label_1'])
    df['user_category' + '_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['user_id', 'category_0','user_category' + '_sum', 'user_category' + '_cvr']], on=['user_id','category_0'], how='left')
    return train_data

def user_item_category_0_hour_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['user_id','category_0', 'hour','is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','category_0', 'hour'], as_index=False).sum()
    df['user_category' + '_hour_sum'] = (df['label_0'] + df['label_1'])  # 历史点击量
    df['user_category' + '_hour_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    train_data = pd.merge(train_data,
                        df[['user_id','category_0', 'hour', 'user_category' + '_hour_sum', 'user_category' + '_hour_cvr']],
                        on=['user_id','category_0', 'hour'], \
                        how='left')
    return train_data

def user_item_category_0_ys_cvr(data,train_data):  # 0.0796573567414
    df = data[data['day'] == 6]
    df = df[['user_id','category_0', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','category_0'], as_index=False).sum()
    df['user_category' + '_sum_ys'] = (df['label_0'] + df['label_1'])
    df['user_category' + '_cvr_ys'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['user_id','category_0', 'user_category' + '_sum_ys', 'user_category' + '_cvr_ys']],
                        on=['user_id','category_0'], how='left')
    return train_data

def user_item_sales_level_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['user_id','item_sales_level', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','item_sales_level'], as_index=False).sum()
    df['user_item_sales' + '_sum'] = (df['label_0'] + df['label_1'])
    df['user_item_sales' + '_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['user_id', 'item_sales_level','user_item_sales' + '_sum', 'user_item_sales' + '_cvr']], on=['user_id','item_sales_level'], how='left')
    return train_data

def user_item_sales_level_hour_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['user_id','item_sales_level', 'hour','is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','item_sales_level', 'hour'], as_index=False).sum()
    df['user_item_sales' + '_hour_sum'] = (df['label_0'] + df['label_1'])  # 历史点击量
    df['user_item_sales' + '_hour_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    train_data = pd.merge(train_data,
                        df[['user_id','item_sales_level', 'hour', 'user_item_sales' + '_hour_sum', 'user_item_sales' + '_hour_cvr']],
                        on=['user_id','item_sales_level', 'hour'], \
                        how='left')
    return train_data

def user_item_sales_level_ys_cvr(data,train_data):  # 0.0796573567414
    df = data[data['day'] == 6]
    df = df[['user_id','item_sales_level', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['user_id','item_sales_level'], as_index=False).sum()
    df['user_item_sales' + '_sum_ys'] = (df['label_0'] + df['label_1'])
    df['user_item_sales' + '_cvr_ys'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['user_id','item_sales_level', 'user_item_sales' + '_sum_ys', 'user_item_sales' + '_cvr_ys']],
                        on=['user_id','item_sales_level'], how='left')
    return train_data

def item_user_gender_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['item_id','user_gender_id', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['item_id','user_gender_id'], as_index=False).sum()
    df['item_user_gender' + '_sum'] = (df['label_0'] + df['label_1'])
    df['item_user_gender' + '_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['item_id', 'user_gender_id','item_user_gender' + '_sum', 'item_user_gender' + '_cvr']], on=['item_id','user_gender_id'], how='left')
    return train_data

def item_user_gender_hour_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['item_id','user_gender_id', 'hour','is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['item_id','user_gender_id', 'hour'], as_index=False).sum()
    df['item_user_gender' + '_hour_sum'] = (df['label_0'] + df['label_1'])  # 历史点击量
    df['item_user_gender' + '_hour_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    train_data = pd.merge(train_data,
                        df[['item_id','user_gender_id', 'hour', 'item_user_gender' + '_hour_sum', 'item_user_gender' + '_hour_cvr']],
                        on=['item_id','user_gender_id', 'hour'], \
                        how='left')
    return train_data

def item_user_gender_ys_cvr(data,train_data):  # 0.0796573567414
    df = data[data['day'] == 6]
    df = df[['item_id','user_gender_id', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['item_id','user_gender_id'], as_index=False).sum()
    df['item_user_gender' + '_sum_ys'] = (df['label_0'] + df['label_1'])
    df['item_user_gender' + '_cvr_ys'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['item_id','user_gender_id', 'item_user_gender' + '_sum_ys', 'item_user_gender' + '_cvr_ys']],
                        on=['item_id','user_gender_id'], how='left')
    return train_data

def item_user_age_level_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['item_id','user_age_level', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['item_id','user_age_level'], as_index=False).sum()
    df['item_user_age' + '_sum'] = (df['label_0'] + df['label_1'])
    df['item_user_age' + '_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['item_id', 'user_age_level','item_user_age' + '_sum', 'item_user_age' + '_cvr']], on=['item_id','user_age_level'], how='left')
    return train_data

def item_user_age_level_hour_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['item_id','user_age_level', 'hour','is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['item_id','user_age_level', 'hour'], as_index=False).sum()
    df['item_user_age' + '_hour_sum'] = (df['label_0'] + df['label_1'])  # 历史点击量
    df['item_user_age' + '_hour_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    train_data = pd.merge(train_data,
                        df[['item_id','user_age_level', 'hour', 'item_user_age' + '_hour_sum', 'item_user_age' + '_hour_cvr']],
                        on=['item_id','user_age_level', 'hour'], \
                        how='left')
    return train_data

def item_user_age_level_ys_cvr(data,train_data):  # 0.0796573567414
    df = data[data['day'] == 6]
    df = df[['item_id','user_age_level', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['item_id','user_age_level'], as_index=False).sum()
    df['item_user_age' + '_sum_ys'] = (df['label_0'] + df['label_1'])
    df['item_user_age' + '_cvr_ys'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['item_id','user_age_level', 'item_user_age' + '_sum_ys', 'item_user_age' + '_cvr_ys']],
                        on=['item_id','user_age_level'], how='left')
    return train_data

def item_user_occupation_id_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['item_id','user_occupation_id', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['item_id','user_occupation_id'], as_index=False).sum()
    df['item_user_occupation' + '_sum'] = (df['label_0'] + df['label_1'])
    df['item_user_occupation' + '_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['item_id', 'user_occupation_id','item_user_occupation' + '_sum', 'item_user_occupation' + '_cvr']], on=['item_id','user_occupation_id'], how='left')
    return train_data

def item_user_occupation_id_hour_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['item_id','user_occupation_id', 'hour','is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['item_id','user_occupation_id', 'hour'], as_index=False).sum()
    df['item_user_occupation' + '_hour_sum'] = (df['label_0'] + df['label_1'])  # 历史点击量
    df['item_user_occupation' + '_hour_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    train_data = pd.merge(train_data,
                        df[['item_id','user_occupation_id', 'hour', 'item_user_occupation' + '_hour_sum', 'item_user_occupation' + '_hour_cvr']],
                        on=['item_id','user_occupation_id', 'hour'], \
                        how='left')
    return train_data

def item_user_occupation_id_ys_cvr(data,train_data):  # 0.0796573567414
    df = data[data['day'] == 6]
    df = df[['item_id','user_occupation_id', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['item_id','user_occupation_id'], as_index=False).sum()
    df['item_user_occupation' + '_sum_ys'] = (df['label_0'] + df['label_1'])
    df['item_user_occupation' + '_cvr_ys'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['item_id','user_occupation_id', 'item_user_occupation' + '_sum_ys', 'item_user_occupation' + '_cvr_ys']],
                        on=['item_id','user_occupation_id'], how='left')
    return train_data

def item_price_user_star_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['item_price_level','user_star_level', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['item_price_level','user_star_level'], as_index=False).sum()
    df['item_price_user_star' + '_sum'] = (df['label_0'] + df['label_1'])
    df['item_price_user_star' + '_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['item_price_level', 'user_star_level','item_price_user_star' + '_sum', 'item_price_user_star' + '_cvr']], on=['item_price_level','user_star_level'], how='left')
    return train_data

def item_price_user_star_hour_cvr(data,train_data):  # 0.0796137691466
    df1 = data[data['day'] < 7]
    df2 = data[data['day'] == 31]
    df = pd.concat([df1, df2],axis = 0,ignore_index = True)
    df = df[['item_price_level','user_star_level', 'hour','is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['item_price_level','user_star_level', 'hour'], as_index=False).sum()
    df['item_price_user_star' + '_hour_sum'] = (df['label_0'] + df['label_1'])  # 历史点击量
    df['item_price_user_star' + '_hour_cvr'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    train_data = pd.merge(train_data,
                        df[['item_price_level','user_star_level', 'hour', 'item_price_user_star' + '_hour_sum', 'item_price_user_star' + '_hour_cvr']],
                        on=['item_price_level','user_star_level', 'hour'], \
                        how='left')
    return train_data

def item_price_user_star_ys_cvr(data,train_data):  # 0.0796573567414
    df = data[data['day'] == 6]
    df = df[['item_price_level','user_star_level', 'is_trade']]
    df = pd.get_dummies(df, columns=['is_trade'], prefix='label')
    df = df.groupby(['item_price_level','user_star_level'], as_index=False).sum()
    df['item_price_user_star' + '_sum_ys'] = (df['label_0'] + df['label_1'])
    df['item_price_user_star' + '_cvr_ys'] = (df['label_1']) / (df['label_0'] + df['label_1'])
    # print(df)
    train_data = pd.merge(train_data, df[['item_price_level','user_star_level', 'item_price_user_star' + '_sum_ys', 'item_price_user_star' + '_cvr_ys']],
                        on=['item_price_level','user_star_level'], how='left')
    return train_data
