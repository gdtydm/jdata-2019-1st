import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os
from tools import reduce_mem_usage
from config import feature_path, base_file_path
import time

'''
    """
    用户表的基础特征， 对缺失值做了填充处理
    # 将数据量小的类别归为一类
    """

用户增长率
其他基础特征，
用户注册距离现在的时间
'''

def label_encoder_2(sum_simple, value_counts):
    bins_encoder_dict = {}
    index = 1
    for k, v in value_counts.items():
        bins_encoder_dict[k] = index
        if v > sum_simple * 0.05:
            index +=1
    return bins_encoder_dict

def extract_user_base_feature(target_date):
    start_time = time.time()
    user_table = pd.read_hdf(base_file_path + "jdata_user.h5")
    user_table['user_reg_tm'] = user_table['user_reg_tm'].apply(lambda s:date(*(int(i) for i in s.split(" ")[0].split("-"))))
    user_table.drop_duplicates(subset="user_id", inplace=True)
    user_table.fillna({"age":-1, "sex":user_table['sex'].median(), "user_lv_cd":user_table['user_lv_cd'].median(), "city_level":user_table['city_level'].median(),
                       "province":user_table['province'].median(), "county":-1, "city":-1}, inplace=True)
    city_map_dict = label_encoder_2(len(user_table), user_table['city'].value_counts())
    user_table['city'] = user_table['city'].apply(lambda s: city_map_dict.get(s))
    county_map_dict = label_encoder_2(len(user_table), user_table['county'].value_counts())
    user_table['county'] = user_table['county'].apply(lambda s: county_map_dict.get(s))
    user_table['reg_to_end_time_day'] = user_table['user_reg_tm'].apply(lambda s: (target_date - s).days)
    user_table.drop("user_reg_tm", axis=1, inplace=True)

    # 会员级别/使用时间
    user_table['consume_rate'] = round(user_table['user_lv_cd'] / user_table['reg_to_end_time_day'] , 2)
    user_table['e_consume_rate'] = np.exp(user_table['consume_rate'])

    user_table = reduce_mem_usage(user_table)
    print(user_table.head())
    print(f"user table shape: {user_table.shape}")
    print(f"extract {str(target_date)} user base feature use：{time.time() - start_time}s")
    # user_table = pd.get_dummies(user_table, columns=['age', "sex", "user_lv_cd", "city_level", "province"])
    user_table.to_hdf(feature_path + f'base_user_feature_{str(target_date)}.h5', key='df', mode='w')

if __name__ == "__main__":
    # 截止时间（<） 不取target_date当天
    for i in [date(2018, 4, 16),date(2018, 4, 9),date(2018, 4, 2)]:
        extract_user_base_feature(i)

