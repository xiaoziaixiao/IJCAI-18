import pandas as pd #数据处理包
import history_feature as hf
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
from sklearn.metrics import log_loss #logloss损失函数H:\JDATA
from sklearn.cross_validation import cross_val_score    #交叉验证
from sklearn.preprocessing import StandardScaler    #标准化数据
from sklearn.preprocessing import MinMaxScaler  #归一化数据
import lightgbm as lgb
#from lightgbm import LGBMClassifier

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

def time_chage(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    data['minute'] = data.time.apply(lambda x: int(x[14:16]))
    x = list(data['item_category_list'].apply(lambda x: x.split(';')))
    x = pd.DataFrame(x, columns=['category_0', 'category_1', 'category_2'])
    x.fillna(value=-1, inplace=True)
    data['category_0'] = x['category_0'].astype('int64')
    data['category_1'] = x['category_1'].astype('int64')
    data['category_2'] = x['category_2'].astype('int64')
    return data

def convert_data(data):
    #基础特征分类,0.0824340892653

    '''''''''''''''''''''''''''''''''''''特征:各种点击click量'''''''''''''''''''''''''''
    user_queue_day = data.groupby(['user_id','day']).size().reset_index().rename(columns = {0 : 'user_day_click'})
    user_queue_day_hour = data.groupby(['user_id','day','hour']).size().reset_index().rename(columns = {0 : 'user_day_hour_click'})

    shop_queue_day = data.groupby(['shop_id','day']).size().reset_index().rename(columns = {0 : 'shop_day_click'})
    shop_queue_day_hour = data.groupby(['shop_id','day','hour']).size().reset_index().rename(columns = {0 : 'shop_day_hour_click'})

    item_queue_day = data.groupby(['item_id', 'day']).size().reset_index().rename(columns={0: 'item_day_click'})  # 0.0801797130137
    item_queue_day_hour = data.groupby(['item_id','day','hour']).size().reset_index().rename(columns = {0:'item_day_hour_click'})#0.0801799262004

    user_item_day_click = data.groupby(['user_id','item_id','day']).size().reset_index().rename(columns = {0 : 'user_item_day_click'})
    user_item_day_hour_click = data.groupby(['user_id', 'item_id', 'day','hour']).size().reset_index().rename(columns={0: 'user_item_day_hour_click'})

    user_shop_day_click = data.groupby(['user_id','shop_id','day']).size().reset_index().rename(columns={0: 'user_shop_day_click'})
    user_shop_day_hour_click = data.groupby(['user_id','shop_id','day','hour']).size().reset_index().rename(columns={0: 'user_shop_day_hour_click'})

    shop_item_day_click = data.groupby(['shop_id', 'item_id', 'day']).size().reset_index().rename(columns={0: 'shop_item_day_click'})  # 0.080342298895
    shop_day_item_click = data.groupby(['shop_id', 'day'])['item_id'].size().reset_index().rename(columns={0: 'shop_day_item_click'})  # 0.0802019214077

    data = pd.merge(data,user_queue_day,on = ['user_id','day'],how = 'left')
    data = pd.merge(data,user_queue_day_hour,on = ['user_id','day','hour'],how = 'left')
    data = pd.merge(data,shop_queue_day,on = ['shop_id','day'],how = 'left')
    data = pd.merge(data,shop_queue_day_hour,on = ['shop_id','day','hour'],how = 'left')
    data = pd.merge(data, item_queue_day, on=['item_id','day'], how='left')
    data = pd.merge(data, item_queue_day_hour, on=['item_id','day','hour'], how='left')

    data = pd.merge(data, user_item_day_click, on=['user_id','item_id','day'], how='left')
    data = pd.merge(data, user_item_day_hour_click, on=['user_id', 'item_id', 'day','hour'], how='left')
    data = pd.merge(data, user_shop_day_click, on=['user_id','shop_id','day'], how='left')
    data = pd.merge(data, user_shop_day_hour_click, on=['user_id', 'shop_id', 'day','hour'], how='left')

    data = pd.merge(data,shop_item_day_click,on = ['shop_id','item_id','day'],how = 'left')
    data = pd.merge(data,shop_day_item_click, on=['shop_id','day'], how='left')

    '''''''''''''''''''''''''''''''''特征：各种标记量，biaoji'''''''''''''''''''''''''''
    '''
    x = data.groupby(['user_id', 'day'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_day_first'})
    data = pd.merge(data, x, on=['user_id', 'day'], how='left')
    data['biaoji_user_day_first'] = (data['context_timestamp'] == data['user_day_first']).apply(lambda x: 1 if x else 0)

    x = data.groupby(['user_id','day'])['context_timestamp'].max().reset_index().rename(columns = {'context_timestamp':'user_day_last'})
    data = pd.merge(data,x,on=['user_id','day'],how = 'left')
    data ['biaoji_user_day_last'] = (data['context_timestamp'] == data['user_day_last']).apply(lambda x: 1 if x else 0)

    x = data.groupby(['shop_id', 'day'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'shop_day_first'})
    data = pd.merge(data, x, on=['shop_id', 'day'], how='left')
    data['biaoji_shop_day_first'] = (data['context_timestamp'] == data['shop_day_first']).apply(lambda x: 1 if x else 0)

    x = data.groupby(['shop_id','day'])['context_timestamp'].max().reset_index().rename(columns = {'context_timestamp':'shop_day_last'})
    data = pd.merge(data,x,on=['shop_id','day'],how = 'left')
    data ['biaoji_shop_day_last'] = (data['context_timestamp'] == data['shop_day_last']).apply(lambda x: 1 if x else 0)

    x = data.groupby(['item_id', 'day'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'item_day_first'})
    data = pd.merge(data, x, on=['item_id', 'day'], how='left')
    data['biaoji_item_day_first'] = (data['context_timestamp'] == data['item_day_first']).apply(lambda x: 1 if x else 0)

    x = data.groupby(['item_id','day'])['context_timestamp'].max().reset_index().rename(columns = {'context_timestamp':'item_day_last'})
    data = pd.merge(data,x,on=['item_id','day'],how = 'left')
    data ['biaoji_item_day_last'] = (data['context_timestamp'] == data['item_day_last']).apply(lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'shop_id', 'day'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_shop_day_first'})
    data = pd.merge(data, x, on=['user_id', 'shop_id', 'day'], how='left')
    data['biaoji_user_shop_day_first'] = (data['context_timestamp'] == data['user_shop_day_first']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'shop_id', 'day'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'user_shop_day_last'})
    data = pd.merge(data, x, on=['user_id', 'shop_id', 'day'], how='left')
    data['biaoji_user_shop_day_last'] = (data['context_timestamp'] == data['user_shop_day_last']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_id', 'day'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_item_day_first'})
    data = pd.merge(data, x, on=['user_id', 'item_id', 'day'], how='left')
    data['biaoji_user_item_day_first'] = (data['context_timestamp'] == data['user_item_day_first']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_id', 'day'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'user_item_day_last'})
    data = pd.merge(data, x, on=['user_id', 'item_id', 'day'], how='left')
    data['biaoji_user_item_day_last'] = (data['context_timestamp'] == data['user_item_day_last']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_brand_id', 'day'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_item_brand_id_day_first'})
    data = pd.merge(data, x, on=['user_id', 'item_brand_id', 'day'], how='left')
    data['biaoji_user_item_brand_id_day_first'] = (data['context_timestamp'] == data['user_item_brand_id_day_first']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_brand_id', 'day'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'user_item_brand_id_day_last'})
    data = pd.merge(data, x, on=['user_id', 'item_brand_id', 'day'], how='left')
    data['biaoji_user_item_brand_id_day_last'] = (data['context_timestamp'] == data['user_item_brand_id_day_last']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_price_level', 'day'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_item_price_level_day_first'})
    data = pd.merge(data, x, on=['user_id', 'item_price_level', 'day'], how='left')
    data['biaoji_user_item_price_level_day_first'] = (
                data['context_timestamp'] == data['user_item_price_level_day_first']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_price_level', 'day'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'user_item_price_level_day_last'})
    data = pd.merge(data, x, on=['user_id', 'item_price_level', 'day'], how='left')
    data['biaoji_user_item_price_level_day_last'] = (
                data['context_timestamp'] == data['user_item_price_level_day_last']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_sales_level', 'day'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_item_sales_level_day_first'})
    data = pd.merge(data, x, on=['user_id', 'item_sales_level', 'day'], how='left')
    data['biaoji_user_item_sales_level_day_first'] = (
                data['context_timestamp'] == data['user_item_sales_level_day_first']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_sales_level', 'day'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'user_item_sales_level_day_last'})
    data = pd.merge(data, x, on=['user_id', 'item_sales_level', 'day'], how='left')
    data['biaoji_user_item_sales_level_day_last'] = (
                data['context_timestamp'] == data['user_item_sales_level_day_last']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'day','hour'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_day_hour_first'})
    data = pd.merge(data, x, on=['user_id', 'day','hour'], how='left')
    data['biaoji_user_day_hour_first'] = (data['context_timestamp'] == data['user_day_hour_first']).apply(lambda x: 1 if x else 0)

    x = data.groupby(['user_id','day','hour'])['context_timestamp'].max().reset_index().rename(columns = {'context_timestamp':'user_day_hour_last'})
    data = pd.merge(data,x,on=['user_id','day','hour'],how = 'left')
    data ['biaoji_user_day_hour_last'] = (data['context_timestamp'] == data['user_day_hour_last']).apply(lambda x: 1 if x else 0)

    x = data.groupby(['shop_id', 'day','hour'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'shop_day_hour_first'})
    data = pd.merge(data, x, on=['shop_id', 'day','hour'], how='left')
    data['biaoji_shop_day_hour_first'] = (data['context_timestamp'] == data['shop_day_hour_first']).apply(lambda x: 1 if x else 0)

    x = data.groupby(['shop_id','day','hour'])['context_timestamp'].max().reset_index().rename(columns = {'context_timestamp':'shop_day_hour_last'})
    data = pd.merge(data,x,on=['shop_id','day','hour'],how = 'left')
    data ['biaoji_shop_day_hour_last'] = (data['context_timestamp'] == data['shop_day_hour_last']).apply(lambda x: 1 if x else 0)

    x = data.groupby(['item_id', 'day','hour'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'item_day_hour_first'})
    data = pd.merge(data, x, on=['item_id', 'day','hour'], how='left')
    data['biaoji_item_day_hour_first'] = (data['context_timestamp'] == data['item_day_hour_first']).apply(lambda x: 1 if x else 0)

    x = data.groupby(['item_id','day','hour'])['context_timestamp'].max().reset_index().rename(columns = {'context_timestamp':'item_day_hour_last'})
    data = pd.merge(data,x,on=['item_id','day','hour'],how = 'left')
    data ['biaoji_item_day_hour_last'] = (data['context_timestamp'] == data['item_day_hour_last']).apply(lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'shop_id','day','hour'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_shop_day_hour_first'})
    data = pd.merge(data, x, on=['user_id','shop_id', 'day','hour'], how='left')
    data['biaoji_user_shop_day_hour_first'] = (data['context_timestamp'] == data['user_shop_day_hour_first']).apply(lambda x: 1 if x else 0)

    x = data.groupby(['user_id','shop_id','day','hour'])['context_timestamp'].max().reset_index().rename(columns = {'context_timestamp':'user_shop_day_hour_last'})
    data = pd.merge(data,x,on=['user_id','shop_id','day','hour'],how = 'left')
    data ['biaoji_user_shop_day_hour_last'] = (data['context_timestamp'] == data['user_shop_day_hour_last']).apply(lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_id', 'day','hour'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_item_day_hour_first'})
    data = pd.merge(data, x, on=['user_id', 'item_id', 'day','hour'], how='left')
    data['biaoji_user_item_day_hour_first'] = (data['context_timestamp'] == data['user_item_day_hour_first']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_id', 'day','hour'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'user_item_day_hour_last'})
    data = pd.merge(data, x, on=['user_id', 'item_id', 'day','hour'], how='left')
    data['biaoji_user_item_day_hour_last'] = (data['context_timestamp'] == data['user_item_day_hour_last']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_brand_id', 'day','hour'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_item_brand_id_day_hour_first'})
    data = pd.merge(data, x, on=['user_id', 'item_brand_id', 'day','hour'], how='left')
    data['biaoji_user_item_brand_id_day_hour_first'] = (
                data['context_timestamp'] == data['user_item_brand_id_day_hour_first']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_brand_id', 'day','hour'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'user_item_brand_id_day_hour_last'})
    data = pd.merge(data, x, on=['user_id', 'item_brand_id', 'day','hour'], how='left')
    data['biaoji_user_item_brand_id_day_hour_last'] = (
                data['context_timestamp'] == data['user_item_brand_id_day_hour_last']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_price_level', 'day','hour'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_item_price_level_day_hour_first'})
    data = pd.merge(data, x, on=['user_id', 'item_price_level', 'day','hour'], how='left')
    data['biaoji_user_item_price_level_day_hour_first'] = (
            data['context_timestamp'] == data['user_item_price_level_day_hour_first']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_price_level', 'day','hour'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'user_item_price_level_day_hour_last'})
    data = pd.merge(data, x, on=['user_id', 'item_price_level', 'day','hour'], how='left')
    data['biaoji_user_item_price_level_day_hour_last'] = (
            data['context_timestamp'] == data['user_item_price_level_day_hour_last']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_sales_level', 'day','hour'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_item_sales_level_day_hour_first'})
    data = pd.merge(data, x, on=['user_id', 'item_sales_level', 'day','hour'], how='left')
    data['biaoji_user_item_sales_level_day_hour_first'] = (
            data['context_timestamp'] == data['user_item_sales_level_day_hour_first']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_sales_level', 'day','hour'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'user_item_sales_level_day_hour_last'})
    data = pd.merge(data, x, on=['user_id', 'item_sales_level', 'day','hour'], how='left')
    data['biaoji_user_item_sales_level_day_hour_last'] = (
            data['context_timestamp'] == data['user_item_sales_level_day_hour_last']).apply(
        lambda x: 1 if x else 0)
    '''
    '''''''''''''''''''''''''''''''''''组合特征，统计量'''''''''''''''''''''''''''''
    '''
    x = data.groupby(['user_id', 'shop_star_level', 'day']).size().reset_index().rename(
        columns={0: 'user_shop_star_click'})
    y = data.groupby(['user_id', 'shop_star_level', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_shop_star_hour_click'})
    data = pd.merge(data, x, on=['user_id', 'shop_star_level', 'day'], how='left')
    data = pd.merge(data, y, on=['user_id', 'shop_star_level', 'day', 'hour'], how='left')
    z = ['user_shop_star_click', 'user_shop_star_hour_click']

    x = data.groupby(['user_id', 'shop_review_num_level', 'day']).size().reset_index().rename(
        columns={0: 'user_shop_review_num_level_click'})
    y = data.groupby(['user_id', 'shop_review_num_level', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_shop_review_num_level_hour_click'})
    data = pd.merge(data, x, on=['user_id', 'shop_review_num_level', 'day'], how='left')
    data = pd.merge(data, y, on=['user_id', 'shop_review_num_level', 'day', 'hour'], how='left')
    z = z + ['user_shop_review_num_level_click', 'user_shop_review_num_level_hour_click']

    x = data.groupby(['user_id', 'shop_review_positive_rate', 'day']).size().reset_index().rename(
        columns={0: 'user_shop_review_positive_rate_click'})
    y = data.groupby(['user_id', 'shop_review_positive_rate', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_shop_review_positive_rate_hour_click'})
    data = pd.merge(data, x, on=['user_id', 'shop_review_positive_rate', 'day'], how='left')
    data = pd.merge(data, y, on=['user_id', 'shop_review_positive_rate', 'day', 'hour'], how='left')
    z = z + ['user_shop_review_positive_rate_click', 'user_shop_review_positive_rate_hour_click']

    x = data.groupby(['user_id', 'shop_score_service', 'day']).size().reset_index().rename(
        columns={0: 'user_shop_score_service_click'})
    y = data.groupby(['user_id', 'shop_score_service', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_shop_score_service_hour_click'})
    data = pd.merge(data, x, on=['user_id', 'shop_score_service', 'day'], how='left')
    data = pd.merge(data, y, on=['user_id', 'shop_score_service', 'day', 'hour'], how='left')
    z = z + ['user_shop_score_service_click', 'user_shop_score_service_hour_click']

    x = data.groupby(['user_id', 'shop_score_delivery', 'day']).size().reset_index().rename(
        columns={0: 'user_shop_score_delivery_click'})
    y = data.groupby(['user_id', 'shop_score_delivery', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_shop_score_delivery_hour_click'})
    data = pd.merge(data, x, on=['user_id', 'shop_score_delivery', 'day'], how='left')
    data = pd.merge(data, y, on=['user_id', 'shop_score_delivery', 'day', 'hour'], how='left')
    z = z + ['user_shop_score_delivery_click', 'user_shop_score_delivery_hour_click']

    x = data.groupby(['user_id', 'shop_score_description', 'day']).size().reset_index().rename(
        columns={0: 'user_shop_score_description_click'})
    y = data.groupby(['user_id', 'shop_score_description', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_shop_score_description_hour_click'})
    data = pd.merge(data, x, on=['user_id', 'shop_score_description', 'day'], how='left')
    data = pd.merge(data, y, on=['user_id', 'shop_score_description', 'day', 'hour'], how='left')
    z = z + ['user_shop_score_description_click', 'user_shop_score_description_hour_click']

    x = data.groupby(['user_id', 'item_brand_id', 'day']).size().reset_index().rename(
        columns={0: 'user_item_brand_id_click'})
    y = data.groupby(['user_id', 'item_brand_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_item_brand_id_hour_click'})
    data = pd.merge(data, x, on=['user_id', 'item_brand_id', 'day'], how='left')
    data = pd.merge(data, y, on=['user_id', 'item_brand_id', 'day', 'hour'], how='left')
    z = z + ['user_item_brand_id_click', 'user_item_brand_id_hour_click']

    x = data.groupby(['user_id', 'item_city_id', 'day']).size().reset_index().rename(
        columns={0: 'user_item_city_id_click'})
    y = data.groupby(['user_id', 'item_city_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_item_city_id_hour_click'})
    data = pd.merge(data, x, on=['user_id', 'item_city_id', 'day'], how='left')
    data = pd.merge(data, y, on=['user_id', 'item_city_id', 'day', 'hour'], how='left')
    z = z + ['user_item_city_id_click', 'user_item_city_id_hour_click']

    x = data.groupby(['user_id', 'item_price_level', 'day']).size().reset_index().rename(
        columns={0: 'user_item_price_click'})
    y = data.groupby(['user_id', 'item_price_level', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_item_price_hour_click'})
    data = pd.merge(data, x, on=['user_id', 'item_price_level', 'day'], how='left')
    data = pd.merge(data, y, on=['user_id', 'item_price_level', 'day', 'hour'], how='left')
    z = z + ['user_item_price_click', 'user_item_price_hour_click']

    x = data.groupby(['user_id', 'item_sales_level', 'day']).size().reset_index().rename(
        columns={0: 'user_item_sales_level_click'})
    y = data.groupby(['user_id', 'item_sales_level', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_item_sales_level_hour_click'})
    data = pd.merge(data, x, on=['user_id', 'item_sales_level', 'day'], how='left')
    data = pd.merge(data, y, on=['user_id', 'item_sales_level', 'day', 'hour'], how='left')
    z = z + ['user_item_sales_level_click', 'user_item_sales_level_hour_click']

    x = data.groupby(['user_id', 'item_collected_level', 'day']).size().reset_index().rename(
        columns={0: 'user_item_collected_level_click'})
    y = data.groupby(['user_id', 'item_collected_level', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_item_collected_level_hour_click'})
    data = pd.merge(data, x, on=['user_id', 'item_collected_level', 'day'], how='left')
    data = pd.merge(data, y, on=['user_id', 'item_collected_level', 'day', 'hour'], how='left')
    z = z + ['user_item_collected_level_click', 'user_item_collected_level_hour_click']

    x = data.groupby(['user_id', 'item_pv_level', 'day']).size().reset_index().rename(columns={0: 'user_item_pv_click'})
    y = data.groupby(['user_id', 'item_pv_level', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_item_pv_hour_click'})
    data = pd.merge(data, x, on=['user_id', 'item_pv_level', 'day'], how='left')
    data = pd.merge(data, y, on=['user_id', 'item_pv_level', 'day', 'hour'], how='left')
    z = z + ['user_item_pv_click', 'user_item_pv_hour_click']

    x = data.groupby(['user_gender_id', 'item_id', 'day']).size().reset_index().rename(columns={0: 'user_gender_item_click'})
    y = data.groupby(['user_gender_id', 'item_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_gender_item_hour_click'})
    data = pd.merge(data, x, on=['user_gender_id', 'item_id', 'day'], how='left')
    data = pd.merge(data, y, on=['user_gender_id', 'item_id', 'day', 'hour'], how='left')
    z = z + ['user_gender_item_click', 'user_gender_item_hour_click']

    x = data.groupby(['user_age_level', 'item_id', 'day']).size().reset_index().rename(columns={0: 'user_age_level_item_click'})
    y = data.groupby(['user_age_level', 'item_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_age_level_item_hour_click'})
    data = pd.merge(data, x, on=['user_age_level', 'item_id', 'day'], how='left')
    data = pd.merge(data, y, on=['user_age_level', 'item_id', 'day', 'hour'], how='left')
    z = z + ['user_age_level_item_click', 'user_age_level_item_hour_click']

    x = data.groupby(['user_occupation_id', 'item_id', 'day']).size().reset_index().rename(columns={0: 'user_occupation_id_item_click'})
    y = data.groupby(['user_occupation_id', 'item_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_occupation_id_item_hour_click'})
    data = pd.merge(data, x, on=['user_occupation_id', 'item_id', 'day'], how='left')
    data = pd.merge(data, y, on=['user_occupation_id', 'item_id', 'day', 'hour'], how='left')
    z = z + ['user_occupation_id_item_click', 'user_occupation_id_item_hour_click']

    x = data.groupby(['user_star_level', 'item_id', 'day']).size().reset_index().rename(columns={0: 'user_star_level_item_click'})
    y = data.groupby(['user_star_level', 'item_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_star_level_item_hour_click'})
    data = pd.merge(data, x, on=['user_star_level', 'item_id', 'day'], how='left')
    data = pd.merge(data, y, on=['user_star_level', 'item_id', 'day', 'hour'], how='left')
    z = z + ['user_star_level_item_click', 'user_star_level_item_hour_click']
    '''
    '''''''''''''''''''新增当天点击率'''''''''
    '''
    data['user_item_dianjilv'] = (data['user_item_day_click'] * 1.0/data['user_day_click'])
    data['user_item_brand_id_dianjilv'] = (data['user_item_brand_id_click'] * 1.0 / data['user_day_click'])
    data['user_item_price_dianjilv'] = (data['user_item_price_click'] * 1.0 / data['user_day_click'])
    data['user_item_sales_level_dianjilv'] = (data['user_item_sales_level_click'] * 1.0 / data['user_day_click'])

    data['user_shop_dianjilv'] = (data['user_shop_day_click'] * 1.0 / data['user_day_click'])
    data['user_shop_star_dianjilv'] = (data['user_shop_star_click'] * 1.0/data['user_day_click'])
    data['user_shop_review_positive_rate_dianjilv'] = (data['user_shop_review_positive_rate_click'] * 1.0 / data['user_day_click'])

    data['user_gender_item_dianjilv'] = (data['user_gender_item_click'] * 1.0 / data['item_day_click'])
    data['user_age_level_item_dianjilv'] = (data['user_age_level_item_click'] * 1.0 / data['item_day_click'])
    data['user_occupation_id_item_dianjilv'] = (data['user_occupation_id_item_click'] * 1.0 / data['item_day_click'])
    data['user_star_level_item_dianjilv'] = (data['user_star_level_item_click'] * 1.0 / data['item_day_click'])
    '''
    '''''''''''''''''''''''''''''''''''标记，点击最多和最少'''''''''''''''''''''''''''''
    '''
    x = data.groupby(['user_id', 'item_id','day'])['user_item_day_click'].max().reset_index().rename(
        columns={'user_item_day_click': 'user_item_click_max'})
    data = pd.merge(data, x, on=['user_id', 'item_id', 'day'], how='left')
    data['biaoji_user_item_click_max'] = (
            data['user_item_day_click'] == data['user_item_click_max']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_id','day'])['user_item_day_click'].min().reset_index().rename(
        columns={'user_item_day_click': 'user_item_click_min'})
    data = pd.merge(data, x, on=['user_id', 'item_id', 'day'], how='left')
    data['biaoji_user_item_click_min'] = (
            data['user_item_day_click'] == data['user_item_click_min']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'shop_id','day'])['user_shop_day_click'].max().reset_index().rename(
        columns={'user_shop_day_click': 'user_shop_click_max'})
    data = pd.merge(data, x, on=['user_id', 'shop_id', 'day'], how='left')
    data['biaoji_user_shop_click_max'] = (
            data['user_shop_day_click'] == data['user_shop_click_max']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'shop_id','day'])['user_shop_day_click'].min().reset_index().rename(
        columns={'user_shop_day_click': 'user_shop_click_min'})
    data = pd.merge(data, x, on=['user_id', 'shop_id', 'day'], how='left')
    data['biaoji_user_shop_click_min'] = (
            data['user_shop_day_click'] == data['user_shop_click_min']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_brand_id','day'])['user_item_brand_id_click'].max().reset_index().rename(
        columns={'user_item_brand_id_click': 'user_item_brand_id_click_max'})
    data = pd.merge(data, x, on=['user_id', 'item_brand_id', 'day'], how='left')
    data['biaoji_user_item_brand_id_click_max'] = (
            data['user_item_brand_id_click'] == data['user_item_brand_id_click_max']).apply(
        lambda x: 1 if x else 0)

    x = data.groupby(['user_id', 'item_brand_id','day'])['user_item_brand_id_click'].min().reset_index().rename(
        columns={'user_item_brand_id_click': 'user_item_brand_id_click_click_min'})
    data = pd.merge(data, x, on=['user_id', 'item_brand_id', 'day'], how='left')
    data['biaoji_user_item_brand_id_click_min'] = (
            data['user_item_brand_id_click'] == data['user_item_brand_id_click_click_min']).apply(
        lambda x: 1 if x else 0)
    '''
    return data

def time_dif(data): #计算时间差，用户当天该次点击距上一次点击和下一次点击的时间差,0.0796398774338
    x = data.sort_index(by = 'context_timestamp')
    x['user_day_diff'] = x.groupby(['user_id','day'])['context_timestamp'].diff(periods = 1)
    x['user_day_diff_next'] = x.groupby(['user_id','day'])['context_timestamp'].diff(periods = -1)
    x['shop_day_diff'] = x.groupby(['shop_id','day'])['context_timestamp'].diff(periods = 1)
    x['shop_day_diff_next'] = x.groupby(['shop_id', 'day'])['context_timestamp'].diff(periods=-1)
    x['item_day_diff'] = x.groupby(['item_id','day'])['context_timestamp'].diff(periods = 1)
    x['item_day_diff_next'] = x.groupby(['item_id', 'day'])['context_timestamp'].diff(periods=-1)
    x['user_item_day_diff'] = x.groupby(['user_id','item_id','day'])['context_timestamp'].diff(periods=1)
    x['user_item_day_diff_next'] = x.groupby(['user_id', 'item_id', 'day'])['context_timestamp'].diff(periods=-1)
    x['user_shop_day_diff'] = x.groupby(['user_id','shop_id','day'])['context_timestamp'].diff(periods=1)
    x['user_shop_day_diff_next'] = x.groupby(['user_id', 'shop_id', 'day'])['context_timestamp'].diff(periods=-1)

    '''
    x['user_day_rank'] = x.groupby(['user_id','day'])['context_timestamp'].rank(method = 'min')
    x['shop_day_rank'] = x.groupby(['shop_id', 'day'])['context_timestamp'].rank(method = 'min')
    x['item_day_rank'] = x.groupby(['item_id', 'day'])['context_timestamp'].rank(method = 'min')
    x['user_item_day_rank'] = x.groupby(['user_id','item_id', 'day'])['context_timestamp'].rank(method = 'min')
    x['user_shop_day_rank'] = x.groupby(['user_id', 'shop_id', 'day'])['context_timestamp'].rank(method='min')

    x['user_day_rank_netx'] = x['user_day_click'] - x['user_day_rank']
    x['shop_day_rank_netx'] = x['shop_day_click'] - x['shop_day_rank']
    x['item_day_rank_netx'] = x['item_day_click'] - x['item_day_rank']
    x['user_item_day_rank_next'] = x['user_item_day_click'] - x['user_item_day_rank']
    x['user_shop_day_rank_next'] = x['user_shop_day_click'] - x['user_shop_day_rank']
    '''
    return x


if __name__ == '__main__':
    feature_base = ['item_id', 'item_brand_id','item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', \
                     'item_pv_level', 'user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id','user_star_level', \
                     'context_id', 'context_timestamp', 'context_page_id','shop_id', 'shop_review_num_level',\
                     'shop_review_positive_rate','shop_star_level', 'shop_score_service', 'shop_score_delivery',\
                     'shop_score_description','day','hour','minute','category_0','category_1','category_2']
    feature_click = ['user_day_click','user_day_hour_click','shop_day_click','shop_day_hour_click','item_day_click',\
                     'item_day_hour_click','user_item_day_click','user_item_day_hour_click',\
                     'user_shop_day_click','user_shop_day_hour_click','shop_item_day_click','shop_day_item_click']
    feature_cvr = ['item_id_sum','item_id_cvr','item_id_hour_sum','item_id_hour_cvr',\
                    'user_id_sum','user_id_cvr','user_id_hour_sum','user_id_hour_cvr',\
                    'shop_id_sum','shop_id_cvr','shop_id_hour_sum','shop_id_hour_cvr',\
                    'user_item_sum','user_item_cvr',\
                    'user_shop_sum','user_shop_cvr', \
                    'user_item_price_sum', 'user_item_price_cvr',\
                    'item_user_gender_sum', 'item_user_gender_cvr',\
                    'user_item_brand_sum','user_item_brand_cvr',\
                    'user_category_sum','user_category_cvr',\
                    'user_item_sales_sum','user_item_sales_cvr',\
                    'item_user_age_sum','item_user_age_cvr',\
                    'item_user_occupation_sum','item_user_occupation_cvr',\
                    'item_price_user_star_sum','item_price_user_star_cvr']
    featuer_cvr1 = ['user_item_hour_sum','user_item_hour_cvr',\
                    'user_shop_hour_sum','user_shop_hour_cvr',\
                    'user_item_price_hour_sum', 'user_item_price_hour_cvr',\
                    'item_user_gender_hour_sum', 'item_user_gender_hour_cvr',\
                    'user_item_brand_hour_sum','user_item_brand_hour_cvr',\
                    'user_category_hour_sum','user_category_hour_cvr',\
                    'user_item_sales_hour_sum','user_item_sales_hour_cvr',\
                    'item_user_age_hour_sum','item_user_age_hour_cvr',\
                    'item_user_occupation_hour_sum','item_user_occupation_hour_cvr',\
                    'item_price_user_star_hour_sum','item_price_user_star_hour_cvr']
    feature_qi = ['item_id_sum_ys','item_id_cvr_ys','item_id_hour_sum','item_id_hour_cvr','user_id_sum_ys','user_id_cvr_ys',\
                    'user_id_hour_sum','user_id_hour_cvr','shop_id_sum_ys','shop_id_cvr_ys',\
                    'shop_id_hour_sum','shop_id_hour_cvr']
    feature_biaoji = ['biaoji_user_day_first','biaoji_user_day_last','biaoji_shop_day_first','biaoji_shop_day_last',\
                      'biaoji_item_day_first','biaoji_item_day_last','biaoji_user_shop_day_first','biaoji_user_shop_day_last', \
                      'biaoji_user_item_day_first', 'biaoji_user_item_day_last',\
                      'biaoji_user_day_hour_first','biaoji_user_day_hour_last','biaoji_shop_day_hour_first',\
                      'biaoji_shop_day_hour_last','biaoji_item_day_hour_first','biaoji_item_day_hour_last',\
                      'biaoji_user_shop_day_hour_first','biaoji_user_shop_day_hour_last',\
                      'biaoji_user_item_day_hour_first', 'biaoji_user_item_day_hour_last']
    feature_biaojim = ['biaoji_user_item_click_max','biaoji_user_item_click_min','biaoji_user_shop_click_max',\
                       'biaoji_user_shop_click_min','biaoji_user_item_brand_id_click_max','biaoji_user_item_brand_id_click_min']

    feature_time_diff = ['user_day_diff','user_day_diff_next','shop_day_diff','shop_day_diff_next','item_day_diff',\
                         'item_day_diff_next','user_item_day_diff','user_item_day_diff_next','user_shop_day_diff',\
                         'user_shop_day_diff_next']
    feature_rank = ['user_day_rank','shop_day_rank','item_day_rank','user_item_day_rank','user_shop_day_rank',\
                     'user_day_rank_netx','shop_day_rank_netx','item_day_rank_netx','user_item_day_rank_next',\
                     'user_shop_day_rank_next']
    target = ['is_trade']
    # 0.171428995413,0.171148331995+z,0.1712+z+tim,0.172471769082,0.171205全特征,0.170593655309新加全特征

    data_train = pd.read_csv('H:/CLProject/[update] round2_ijcai_18_train_20180425/round2_train.txt',sep = ' ')
    data_test = pd.read_csv('H:/CLProject/round2_ijcai_18_test_a_20180425/round2_ijcai_18_test_a_20180425.txt', sep=' ')
    data_test_0 = pd.read_csv('H:/round2_ijcai_18_test_b_20180510/round2_ijcai_18_test_b_20180510.txt', sep=' ')
    print(data_test_0.shape)
    X_test_instance_id_co = data_test_0['instance_id'].values
    data_test['is_trade'] = -1
    data_test_0['is_trade'] = -1

    data =  pd.concat([data_train,data_test,data_test_0],axis = 0,ignore_index = True)
    data['is_trade'] = data['is_trade'].astype('int')
    train_data = time_chage(data)
    print(train_data.shape, train_data.columns)
    #train_data = data

    begin = time.time()
    #train_data = data[data['day'] == 7]
    train_data = convert_data(train_data)
    print('1',(time.time()-begin)/60,train_data.shape,train_data.columns)

    begin = time.time()
    train_data = hf.item_cvr(data,train_data)
    #train_data = hf.item_ys_cvr(data,train_data)
    train_data = hf.item_hour_cvr(data,train_data)
    train_data = hf.user_cvr(data,train_data)
    train_data = hf.user_hour_cvr(data,train_data)
    #train_data = hf.user_ys_cvr(data,train_data)
    train_data = hf.shop_cvr(data,train_data)
    train_data = hf.shop_hour_cvr(data,train_data)
    #train_data = hf.shop_ys_cvr(data,train_data)

    train_data = hf.user_item_cvr(data,train_data)
    train_data = hf.user_shop_cvr(data, train_data)
    train_data = hf.user_item_brand_cvr(data,train_data)
    train_data = hf.user_item_price_cvr(data, train_data)
    train_data = hf.user_item_category_0_cvr(data, train_data)
    train_data = hf.user_item_sales_level_cvr(data, train_data)
    train_data = hf.item_user_gender_cvr(data, train_data)
    train_data = hf.item_user_age_level_cvr(data, train_data)
    train_data = hf.item_user_occupation_id_cvr(data, train_data)
    train_data = hf.item_price_user_star_cvr(data, train_data)

    train_data = hf.user_item_ys_cvr(data,train_data)
    train_data = hf.user_shop_ys_cvr(data, train_data)
    train_data = hf.user_item_brand_ys_cvr(data,train_data)
    train_data = hf.user_item_price_ys_cvr(data, train_data)
    train_data = hf.user_item_category_0_ys_cvr(data, train_data)
    train_data = hf.user_item_sales_level_ys_cvr(data, train_data)
    train_data = hf.item_user_gender_ys_cvr(data, train_data)
    train_data = hf.item_user_age_level_ys_cvr(data, train_data)
    train_data = hf.item_user_occupation_id_ys_cvr(data, train_data)
    train_data = hf.item_price_user_star_ys_cvr(data, train_data)

    train_data = hf.user_item_hour_cvr(data,train_data)
    train_data = hf.user_shop_hour_cvr(data, train_data)
    train_data = hf.user_item_brand_hour_cvr(data,train_data)
    train_data = hf.user_item_price_hour_cvr(data, train_data)
    train_data = hf.user_item_category_0_hour_cvr(data, train_data)
    train_data = hf.user_item_sales_level_hour_cvr(data, train_data)
    train_data = hf.item_user_gender_hour_cvr(data, train_data)
    train_data = hf.item_user_age_level_hour_cvr(data, train_data)
    train_data = hf.item_user_occupation_id_hour_cvr(data, train_data)
    train_data = hf.item_price_user_star_hour_cvr(data, train_data)

    print('2',(time.time() - begin) / 60, train_data.shape, train_data.columns)
    

    feature_biaoji = ['biaoji_user_day_first','biaoji_shop_day_first',\
                      'biaoji_item_day_first','biaoji_user_shop_day_first', \
                      'biaoji_user_item_day_first']
    feature_click = ['user_day_click', 'shop_day_click','item_day_click', 'user_item_day_click','user_shop_day_click', \
                      'shop_item_day_click']
    feature_cvr = ['user_item_sum','user_item_cvr','user_shop_sum','user_shop_cvr', 'user_item_price_sum','user_item_price_cvr',\
                   'item_user_gender_sum','item_user_gender_cvr']

    feature_zuhe = ['user_shop_star_click','user_shop_star_hour_click',\
                    'user_shop_review_positive_rate_click','user_shop_review_positive_rate_hour_click',\
                    'user_item_brand_id_click','user_item_brand_id_hour_click',\
                    'user_item_price_click','user_item_price_hour_click',\
                    'user_item_sales_level_click','user_item_sales_level_hour_click',\
                    'user_gender_item_click','user_gender_item_hour_click',\
                    'user_age_level_item_click','user_age_level_item_hour_click',\
                    'user_occupation_id_item_click','user_occupation_id_item_hour_click',\
                    'user_star_level_item_click','user_star_level_item_hour_click']
    feature_dianjilv = ['user_item_dianjilv','user_item_brand_id_dianjilv','user_item_price_dianjilv','user_item_sales_level_dianjilv',\
                        'user_shop_dianjilv','user_shop_star_dianjilv','user_shop_review_positive_rate_dianjilv',\
                        'user_gender_item_dianjilv','user_age_level_item_dianjilv','user_occupation_id_item_dianjilv','user_star_level_item_dianjilv']

    begin = time.time()
    train_data = time_dif(train_data)
    print((time.time() - begin) / 60, train_data.shape, train_data.columns)

    #feature = [i for i in feature if i not in no_feature]
    feature = feature_base + feature_click + feature_biaoji + feature_cvr  + feature_time_diff + feature_zuhe + feature_dianjilv

    feature = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level','item_collected_level', \
               'item_pv_level', 'user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id','user_star_level', \
               'context_id', 'context_timestamp', 'context_page_id', 'shop_id', 'shop_review_num_level', \
               'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', \
               'shop_score_description', 'day', 'hour', 'minute', 'category_0', 'category_1', 'category_2']
    '''
    #feature = feature_base
    #feature = [i for i in feature if i not in feature_qi]
    #0.177132207191,0.175816371715,(0.17300203792,0.172628962166),(0.17173909582 3,0.17176482939),0.17179447769,(0.170680808142,0.170654949124不加rank,0.170658331214调参)
    #0.171890987737,0.17185119113,0.171860653554,0.171875863716,(0.172663,0.17264)
    # (0.170064,新增转化率-0.16991029084,新增组合小时点击-0.169925343636)，不家组合-0.169914714415,不家小时-0.16991029084
    #全特征-0.170025398691,0.170105900394,不加点击率-0.17017053898，去标记-0.169926698868，去标记+点击率-0.169860307721,去小时标记-0.170078978847
    #0.169860307721,去掉小时和昨天的历史-0.169921700403，去掉组合-0.169865487047
    #基础特征X-0.097447，LGBM-0.1696，加历史小时X-0.169938155257,加历史小时L-0.169748279132,基础X不加参-0.0986312555759
    #去掉昨天历史-0.1697，基础-0.176525793493,去掉cvr1-0.169677998916，去掉点击率-0.169835809681
    test_data_1 = train_data[train_data['hour'] >=12]
    train_data_1 = train_data[train_data['hour'] < 12]

    test_data = train_data[(train_data['hour'] > 7) & (train_data['hour'] < 12)]
    train_data = train_data[train_data['hour'] <= 7]
    '''
    '''
    X_train = train_data[feature]
    y_train = train_data[target]
    X_test = test_data[feature]
    y_test = test_data[target]
    print(X_train.shape,X_test.shape)
    print(X_train.columns,X_test.columns)
    
    begin = time.time()
    vec = DictVectorizer(sparse = False)
    X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
    X_test = vec.fit_transform(X_test.to_dict(orient = 'record'))
    print('3',(time.time() - begin) / 60)
    
    xgbc = XGBClassifier(max_depth = 5,n_estimators = 1000, learning_rate=0.05)#0.171762705738,0.172061，0.172059，0.172092
    xgbc.fit(X_train,y_train,eval_set = [(X_test,y_test)],early_stopping_rounds = 30,verbose = 1,eval_metric='logloss' )
    xgbc_predict = xgbc.predict(X_test)
    xgbc_predict_prob = xgbc.predict_proba(X_test)
    print(log_loss(y_test,xgbc_predict_prob)
    
    lgbc = lgb.LGBMClassifier(num_leaves = 100,max_depth = 5,n_estimators = 10000, learning_rate=0.01)#0.171762705738,0.172061，0.172059，0.172092
    lgbc.fit(X_train,y_train,eval_set = [(X_test,y_test)],early_stopping_rounds = 30,verbose = 1,eval_metric='logloss' )
    xgbc_predict = lgbc.predict(X_test)
    xgbc_predict_prob = lgbc.predict_proba(X_test)
    print(log_loss(y_test,xgbc_predict_prob))
    '''
    '''
    begin = time.time()
    X_train = train_data_1[feature]
    y_train = train_data_1[target]
    X_test = test_data_1[feature]
    print(X_train.shape, X_test.shape)

    X_test_instance_id = test_data_1['instance_id'].values

    vec = DictVectorizer(sparse=False)
    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = vec.fit_transform(X_test.to_dict(orient='record'))
    
    xgbc = XGBClassifier(max_depth = 5,n_estimators = 1000, learning_rate=0.05)
    xgbc.fit(X_train, y_train)
    xgbc_predict = xgbc.predict_proba(X_test)
    print(xgbc_predict)
    
    lgbc = lgb.LGBMClassifier(num_leaves = 100,max_depth = 5,n_estimators = 10000, learning_rate=0.01)#0.171762705738,0.172061，0.172059，0.172092
    lgbc.fit(X_train,y_train)
    xgbc_predict = lgbc.predict_proba(X_test)
    print(xgbc_predict)
    
    # result = pd.concat([X_test_instance_id_co,xgbc_predict[:1]],axis = 1)
    result = {}
    for i in range(len(xgbc_predict)):
        result[X_test_instance_id[i]] = xgbc_predict[i][1]
    print(result)

    with open('H:/CLProject/result/result-d-7.txt', 'a+') as f:
        f.write('instance_id predicted_score\n')
    for i in X_test_instance_id_co:
        print('%s %f\n' % (i, result[i]))
        with open('H:/CLProject/result/result-d-7.txt', 'a+') as f:
            f.write('%s %f\n' % (i, result[i]))
    print('4',(time.time() - begin) / 60)
    '''
